"""Fixed protocol: honest cell-level error prediction, then gated normalized CP."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.fft import dct
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_t100 as base

METHODS = ['simple', 'embedding', 'multiscale']
SEEDS = [42, 2021, 2024]
CFG = SimpleNamespace(d_model=128, d_ff=256, e_layers=6, d_layers=4,
                      n_heads=4, lr=5e-5, batch_size=16, epochs=100, patience=10)


def save_json(path, obj):
    def convert(v):
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, Path): return str(v)
        raise TypeError(type(v).__name__)
    path.write_text(json.dumps(obj, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def upstream_reader(root):
    """Execute original preprocessing methods without importing unrelated packages."""
    source = base.UPSTREAM / 'data_provider/data_loader.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Dataset_original')
    wanted = {'read_cell_df', 'get_charge_discharge_curves', 'resample_charge_discharge_curves'}
    module = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
    reader_cls = type('OriginalMethods', (), {name: namespace[name] for name in wanted})
    reader = reader_cls()
    reader.early_cycle_threshold = 100
    reader.charge_discharge_len = 300
    reader.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah',
                        'discharge_capacity_in_Ah', 'time_in_s']
    reader.ZN_coin_charge_first_file_names = []
    # Augmentation outputs do not affect the original unaugmented input.
    reader.aug_helper = SimpleNamespace(batch_aug=lambda x: (x, x))
    return reader


def load_data(root, names):
    labels_path = root / 'Life labels/MATR_labels.json'
    labels = json.loads(labels_path.read_text())
    source_hash = hashlib.sha256((Path(base.__file__).read_bytes() +
        (base.UPSTREAM / 'data_provider/data_loader.py').read_bytes())).hexdigest()[:16]
    cache = HERE / 'cache' / source_hash
    cache.mkdir(parents=True, exist_ok=True)
    reader = upstream_reader(root)
    xs, ys, audit = [], [], []
    for i, name in enumerate(names):
        path = root / 'MATR' / name
        if name not in labels or not path.exists():
            raise ValueError(f'Missing source/label for {name}; do not silently change fixed splits')
        eol = float(labels[name])
        if not np.isfinite(eol) or eol <= 100: raise ValueError(f'Ineligible EOL: {name}')
        stat = path.stat()
        key = hashlib.sha256(f'{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{eol}'.encode()).hexdigest()[:16]
        target = cache / f'{path.stem}_{key}.npz'
        if target.exists():
            with np.load(target) as record:
                x, delta = record['x'], float(record['max_abs_difference'])
        else:
            x = base.load_cell(path, eol)
            with path.open('rb') as f: raw = pickle.load(f)
            reader.read_cell_data_according_to_prefix = lambda _, raw=raw, eol=eol: (raw, eol)
            original = reader.read_cell_df(name)[1].astype(np.float32)
            delta = float(np.max(np.abs(x - original)))
            np.testing.assert_allclose(x, original, rtol=1e-6, atol=1e-6, err_msg=name)
            np.savez_compressed(target, x=x, max_abs_difference=delta)
        assert x.shape == (100, 3, 300) and np.isfinite(x).all()
        xs.append(x); ys.append(eol)
        audit.append({'cell_id': path.stem, 'max_abs_difference': delta})
        if (i + 1) % 20 == 0: print(f'data {i+1}/{len(names)} verified', flush=True)
    return np.stack(xs), np.asarray(ys), audit


def features(x):
    series = x.astype(np.float64).mean(axis=-1)  # cell, cycle, channel
    t = np.arange(100, dtype=float) - 49.5
    slope = np.einsum('ntc,t->nc', series, t) / np.sum(t*t)
    simple = np.concatenate([series[:, :10].mean(1), series[:, -10:].mean(1),
                             slope, series.std(1)], axis=1)
    spectrum = dct(series - series.mean(1, keepdims=True), axis=1, norm='ortho')
    energy = np.concatenate([np.mean(spectrum[:, a:b]**2, axis=1)
                             for a, b in [(1, 6), (6, 21), (21, 100)]], axis=1)
    return simple, np.concatenate([simple, energy], axis=1)


def partitions(names):
    folds = []
    for fold, (remaining, evaluation) in enumerate(KFold(5, shuffle=True, random_state=2024).split(names)):
        pool = np.random.default_rng(2024 + fold).permutation(remaining)
        error_train, predictor = pool[:20], pool[20:]
        folds.append({'predictor': predictor.tolist(), 'error_train': error_train.tolist(),
                      'evaluation': evaluation.tolist(), 'validation': predictor[:8].tolist(),
                      'selection_train': predictor[8:].tolist()})
        assert not (set(predictor) & set(error_train) or set(predictor) & set(evaluation)
                    or set(error_train) & set(evaluation))
    assert sorted(i for f in folds for i in f['evaluation']) == list(range(len(names)))
    return folds


def train_predictor(x, y, fit_ids, val_ids, seed, device, fixed_epochs=None):
    base.seed_all(seed)
    model = base.Model(base.model_config(CFG)).to(device)
    mean, scale = float(y[fit_ids].mean()), float(y[fit_ids].std())
    assert scale > 0
    loader = DataLoader(TensorDataset(torch.from_numpy(x[fit_ids]),
        torch.tensor((y[fit_ids]-mean)/scale, dtype=torch.float32)),
        batch_size=CFG.batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    history, best, best_epoch, stale = [], float('inf'), 0, 0
    for epoch in range(1, (fixed_epochs or CFG.epochs)+1):
        model.train()
        total, count = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb, torch.ones((len(xb), 100), device=device)).squeeze(-1)
            loss = torch.mean((pred-yb)**2)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            loss.backward(); opt.step()
            total += loss.item()*len(xb); count += len(xb)
        row = {'epoch': epoch, 'train_scaled_mse': total/count}
        if fixed_epochs is None:
            vp = base.infer(model, x[val_ids], device, CFG.batch_size)[0]*scale+mean
            vmape = float(np.mean(np.abs(y[val_ids]-vp)/y[val_ids]))
            row['validation_mape'] = vmape
            if vmape < best: best, best_epoch, stale = vmape, epoch, 0
            else: stale += 1
        history.append(row)
        if fixed_epochs is None and stale >= CFG.patience: break
    return model, mean, scale, best_epoch, history


def fit_and_predict(x, y, split, seed, device, out):
    out.mkdir(parents=True, exist_ok=True)
    bundle = out / 'frozen_predictions.npz'
    if bundle.exists():
        with np.load(bundle) as r: return r['pred'], r['embedding']
    selection, _, _, epoch, history = train_predictor(x, y, split['selection_train'],
        split['validation'], seed, device)
    del selection
    model, mean, scale, _, refit_history = train_predictor(x, y, split['predictor'],
        [], seed, device, fixed_epochs=epoch)
    pred, embedding = base.infer(model, x, device, CFG.batch_size, embeddings=True)
    pred = pred.astype(float)*scale+mean
    assert np.isfinite(pred).all() and np.isfinite(embedding).all()
    torch.save({'state_dict': model.state_dict(), 'label_mean': mean, 'label_scale': scale,
                'epoch': epoch}, out / 'model.pt')
    save_json(out / 'training.json', {'selected_epoch': epoch, 'selection': history, 'refit': refit_history})
    np.savez_compressed(bundle, pred=pred, embedding=embedding)
    del model
    print(f'{out.name}: selected epoch {epoch}', flush=True)
    return pred, embedding


def fit_error_models(x, y, pred, embedding, split, seed, out):
    p, e = split['predictor'], split['error_train']
    simple, multi = features(x)
    # PCA coordinates are shared within a fold, never pooled across predictors.
    pca = PCA(n_components=8, svd_solver='full').fit(embedding[p])
    inputs = {'simple': np.column_stack([pred, simple]),
              'embedding': np.column_stack([pred, pca.transform(embedding)]),
              'multiscale': np.column_stack([pred, multi])}
    target = np.log1p(np.abs(y[e]-pred[e]))
    logs = {'constant': np.full(len(x), target.mean())}
    fitted = {}
    for name in METHODS:
        reg = make_pipeline(StandardScaler(), Ridge(alpha=10))
        reg.fit(inputs[name][e], target)
        logs[name] = reg.predict(inputs[name])
        fitted[name] = reg
    joblib.dump({'pca': pca, 'models': fitted, 'constant': target.mean()}, out / 'error_models.joblib')
    return logs


def diagnostic_metrics(rows):
    y, pred = rows.eol.to_numpy(), rows.pred.to_numpy()
    error = np.abs(y-pred); target = np.log1p(error)
    baseline = float(np.mean((target-rows['constant'])**2))
    result = {'point': {name: base.point_metrics(y, rows[col].to_numpy())
                        for name, col in [('CPTransformer', 'pred'), ('Dummy', 'dummy'), ('Ridge', 'ridge')]},
              'errors': {}}
    for name in ['constant']+METHODS:
        values = rows[name].to_numpy()
        mse = float(np.mean((target-values)**2))
        rho = float(spearmanr(error, values).statistic) if np.ptp(values)>0 else None
        high = rows[f'{name}_high'].to_numpy(dtype=bool)
        g = np.maximum(1., np.expm1(values))
        if not np.isfinite(g).all(): raise FloatingPointError('Nonfinite error scale')
        result['errors'][name] = {'log_mse': mse, 'improvement': 1-mse/baseline,
            'spearman': rho, 'absolute_error_mae': float(np.mean(np.abs(error-g))),
            'high_risk_actual_mae': float(error[high].mean()) if high.any() else None,
            'low_risk_actual_mae': float(error[~high].mean()) if (~high).any() else None,
            'high_risk_n': int(high.sum())}
    return result


def diagnose(x, y, names, folds, seed, device, out):
    seed_dir = out / f'seed_{seed}'
    seed_dir.mkdir(exist_ok=True)
    parts = []
    simple = features(x)[0]
    for k, split in enumerate(folds):
        fold_dir = seed_dir / f'fold_{k}'
        pred, emb = fit_and_predict(x, y, split, seed, device, fold_dir)
        logs = fit_error_models(x, y, pred, emb, split, seed, fold_dir)
        ids, p, e = split['evaluation'], split['predictor'], split['error_train']
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(simple[p], y[p])
        rows = pd.DataFrame({'cell_id': [names[i][:-4] for i in ids], 'fold': k,
            'eol': y[ids], 'pred': pred[ids], 'dummy': y[p].mean(), 'ridge': ridge.predict(simple[ids])})
        for name, values in logs.items():
            rows[name] = values[ids]
            rows[f'{name}_high'] = values[ids] > np.median(values[e])
        rows.to_csv(fold_dir / 'evaluation.csv', index=False)
        parts.append(rows)
        print(f'seed {seed} fold {k+1}/5 complete', flush=True)
    rows = pd.concat(parts, ignore_index=True)
    assert len(rows) == len(names) and rows.cell_id.nunique() == len(names)
    rows.to_csv(seed_dir / 'evaluation.csv', index=False)
    metrics = diagnostic_metrics(rows)
    save_json(seed_dir / 'metrics.json', metrics)
    print(f'SEED {seed}: '+json.dumps(metrics), flush=True)
    return metrics


def choose(all_metrics):
    results = {}
    for name in METHODS:
        vals = [m['errors'][name] for m in all_metrics]
        improvement = float(np.mean([v['improvement'] for v in vals]))
        positive = sum(v['improvement'] > 0 and v['spearman'] is not None and v['spearman'] > 0 for v in vals)
        results[name] = {'mean_improvement': improvement, 'positive_seeds': positive,
                         'passes': improvement >= .05 and positive >= 2}
    passed = [n for n in METHODS if results[n]['passes']]
    selected = None
    if passed:
        best = max(results[n]['mean_improvement'] for n in passed)
        selected = next(n for n in passed if best-results[n]['mean_improvement'] < .01)
    return {'candidates': results, 'selected': selected,
            'rule': 'mean relative log-MSE improvement >=5%; >=2 seeds improve and have positive Spearman'}


def intervals(root, x, y, names, selected, device, out):
    extra_names = list(base.split_recorder.MATR_val_files)+list(base.split_recorder.MATR_test_files)
    assert not set(names) & set(extra_names) and len(set(extra_names)) == len(extra_names)
    extra_x, extra_y, audit = load_data(root, extra_names)
    x, y = np.concatenate([x, extra_x]), np.concatenate([y, extra_y])
    all_names = names+extra_names
    pool = np.random.default_rng(2024).permutation(len(names))
    p, e = pool[20:].tolist(), pool[:20].tolist()
    split = {'predictor': p, 'error_train': e, 'selection_train': p[8:], 'validation': p[:8]}
    cal = list(range(len(names), len(names)+len(base.split_recorder.MATR_val_files)))
    test = list(range(cal[-1]+1, len(all_names)))
    save_json(out / 'interval_split.json', {**{k:[all_names[i] for i in v] for k,v in split.items()},
        'calibration': [all_names[i] for i in cal], 'test':[all_names[i] for i in test], 'audit':audit})
    summary = {}
    for seed in SEEDS:
        folder = out / f'interval_seed_{seed}'
        pred, emb = fit_and_predict(x, y, split, seed, device, folder)
        logs = fit_error_models(x, y, pred, emb, split, seed, folder)
        g = np.maximum(1., np.expm1(logs[selected]))
        if not np.isfinite(g).all(): raise FloatingPointError('Nonfinite g')
        r = np.abs(y[cal]-pred[cal])
        qa, rank = base.conformal_upper_quantile(r, .1)
        qn, _ = base.conformal_upper_quantile(r/g[cal], .1)
        rows = pd.DataFrame({'cell_id':[all_names[i][:-4] for i in test], 'eol':y[test], 'pred':pred[test], 'g':g[test]})
        metrics = {'point':base.point_metrics(y[test], pred[test]), 'method':selected, 'rank':rank,
                   'historical_test_previously_viewed':True}
        for name, radius in [('split',np.full(len(test),qa)), ('normalized',qn*g[test])]:
            lo, hi = pred[test]-radius, pred[test]+radius
            rows[name+'_lower'], rows[name+'_upper'] = lo, hi
            metrics[name] = {**base.interval_metrics(y[test],lo,hi,.1),
                'miss_lower':int(np.sum(y[test]<lo)), 'miss_upper':int(np.sum(y[test]>hi)),
                'q':qa if name=='split' else qn}
        rows.to_csv(folder / 'test_predictions.csv', index=False)
        pd.DataFrame({'cell_id':[all_names[i][:-4] for i in cal], 'eol':y[cal],
                      'pred':pred[cal], 'g':g[cal]}).to_csv(folder / 'calibration_predictions.csv', index=False)
        save_json(folder / 'metrics.json', metrics)
        summary[str(seed)] = metrics
        print(f'INTERVAL SEED {seed}: '+json.dumps(metrics), flush=True)
    save_json(out / 'interval_summary.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path(r'D:\BatteryLife_Processed'))
    parser.add_argument('--resume', type=Path, help='Resume only a run created by this exact implementation')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = {'predictor':vars(CFG), 'split_seed':2024, 'model_seeds':SEEDS, 'device':device,
              'data_root':str(args.data_root.resolve()), 'code_sha256':code_hash,
              'torch':torch.__version__, 'numpy':np.__version__, 'ridge_alpha':10, 'pca_dim':8,
              'scale_floor':1., 'dct_norm':'ortho', 'gate_improvement':.05, 'gate_positive_seeds':2}
    if args.resume:
        out = args.resume.resolve()
        old = json.loads((out/'config.json').read_text())
        if old != config: raise ValueError('Resume configuration/source mismatch')
    else:
        out = HERE / 'runs' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        out.mkdir(parents=True, exist_ok=False)
        save_json(out / 'config.json', config)
    print(f'RUN_DIR={out}', flush=True)
    names = list(base.split_recorder.MATR_train_files)
    assert len(names)==102 and len(set(names))==102
    x, y, audit = load_data(args.data_root, names)
    save_json(out / 'data_audit.json', audit)
    folds = partitions(names)
    save_json(out / 'splits.json', [{k:[names[i] for i in ids] for k,ids in f.items()} for f in folds])
    metrics = [diagnose(x,y,names,folds,seed,device,out) for seed in SEEDS]
    decision = choose(metrics)
    save_json(out / 'decision.json', decision)
    print('DECISION: '+json.dumps(decision), flush=True)
    if decision['selected'] is not None:
        intervals(args.data_root,x,y,names,decision['selected'],device,out)
    else:
        print('Gate not passed; interval experiments stopped by protocol.', flush=True)


if __name__ == '__main__':
    main()
