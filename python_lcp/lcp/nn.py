"""
Neural network modules for LCP experiments.

Translates my_nn_modules.R — PyTorch-based NN training with CV support.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Callable
import copy


# ── Neural network architecture ─────────────────────────────────────────────

class MyNeuralNet(nn.Module):
    """Equivalent to R my_neuralnet nn_module."""

    def __init__(self, in_shape: int = 1, out_shape: int = 1,
                 hidden_size: int = 64, drop_out: float = 0.5):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.hidden_size = hidden_size
        self.drop_out = drop_out

        self.base_model = nn.Sequential(
            nn.Linear(in_shape, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_size, out_shape),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)


# ── Loss functions ──────────────────────────────────────────────────────────

class AllQuantileLoss(nn.Module):
    """Quantile loss (pinball loss) for multiple quantiles.
    Equivalent to R all_quantile_loss nn_module."""

    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (batch, num_quantiles), target: (batch, 1) or (batch,)
        if target.dim() == 1:
            target = target.unsqueeze(1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target[:, 0] - preds[:, i]
            loss_i = torch.max((q - 1) * errors, q * errors)
            losses.append(loss_i.unsqueeze(1))
        losses = torch.cat(losses, dim=1)
        return torch.mean(torch.sum(losses, dim=1))


# ── Data splitting ──────────────────────────────────────────────────────────

def train_test_split(x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2,
                     random_state: Optional[int] = None) -> dict:
    """Equivalent to R my_test_split_func()."""
    if random_state is not None:
        np.random.seed(random_state)
    n = len(y)
    n_test = int(np.ceil(n * test_ratio))
    idx = np.random.permutation(n)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return {
        "xtrain": x[train_idx], "ytrain": y[train_idx],
        "xtest": x[test_idx], "ytest": y[test_idx],
    }


def cv_split(x: np.ndarray, y: np.ndarray, nfolds: int = 10,
             random_state: Optional[int] = None) -> dict:
    """Equivalent to R my_cv_split_func()."""
    if random_state is not None:
        np.random.seed(random_state)
    n = len(y)
    shuffled = np.random.permutation(n)
    n0 = int(np.ceil(n / nfolds))
    k01 = nfolds - (n0 * nfolds - n)

    xtrain_list, ytrain_list = [], []
    xtest_list, ytest_list = [], []
    foldid_list = []

    idx_start = 0
    for k in range(nfolds):
        fold_size = n0 if k < k01 else n0 - 1
        fold_idx = shuffled[idx_start:idx_start + fold_size]
        idx_start += fold_size

        foldid_list.append(fold_idx)
        xtest_list.append(x[fold_idx])
        ytest_list.append(y[fold_idx])

        train_mask = np.ones(n, dtype=bool)
        train_mask[fold_idx] = False
        xtrain_list.append(x[train_mask])
        ytrain_list.append(y[train_mask])

    return {
        "xtrain": xtrain_list, "ytrain": ytrain_list,
        "xtest": xtest_list, "ytest": ytest_list,
        "foldid": foldid_list,
    }


# ── Jacobian computation ────────────────────────────────────────────────────

def compute_jacobian(inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """Compute Jacobian of each output dimension w.r.t. inputs via autograd.
    Equivalent to R compute_jacobian_autograd()."""
    num_classes = outputs.shape[1]
    jacobian = torch.zeros(num_classes, inputs.shape[0], inputs.shape[1],
                           device=inputs.device)
    grad_output = torch.zeros(outputs.shape[0], outputs.shape[1],
                              device=inputs.device)

    for i in range(num_classes):
        if inputs.grad is not None:
            inputs.grad.zero_()
        grad_output.zero_()
        grad_output[:, i] = 1.0
        outputs.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad.data.clone()

    return jacobian.transpose(0, 1)


# ── One-epoch training ──────────────────────────────────────────────────────

def epoch_train(model: nn.Module, loss_func: callable,
                xtrain: np.ndarray, ytrain: np.ndarray,
                batch_size: int, optimizer_class,
                lr: float, wd: float, device: torch.device,
                cnt: int = 0, best_cnt: float = float('inf')) -> dict:
    """Equivalent to R my_epoch_internal_train()."""
    model = model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=wd)
    model.train()

    n = xtrain.shape[0]
    shuffle_idx = np.random.permutation(n)
    xtrain = xtrain[shuffle_idx]
    ytrain = ytrain[shuffle_idx]

    xt = torch.tensor(xtrain, dtype=torch.float32, device=device)
    yt = torch.tensor(ytrain, dtype=torch.float32, device=device)

    epoch_losses = []
    n_batches = int(np.ceil(n / batch_size))

    for idx in range(n_batches):
        cnt += 1
        start = idx * batch_size
        end = min(start + batch_size, n)
        batch_x = xt[start:end]
        batch_y = yt[start:end]

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = loss_func(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

        if cnt >= best_cnt:
            break

    return {
        "model": model,
        "epoch_loss": float(np.mean(epoch_losses)),
        "cnt": cnt,
    }


# ── Learner with CV support ─────────────────────────────────────────────────

class LearnerOptimizer:
    """Equivalent to R learnerOptimizer R6 class.

    Manages NN training with cross-validation, checkpointing, and prediction
    with Jacobian computation.
    """

    def __init__(
        self,
        network_factory: Callable[..., nn.Module],
        optimizer_class,
        loss_func: callable,
        device: str = "cpu",
        loss_type: str = "mse",
        all_quantiles: Optional[List[float]] = None,
        test_ratio: float = 0.2,
        CV: bool = True,
        nfolds: int = 5,
        random_state: int = 0,
        hidden_size: int = 64,
        in_shape: int = 1,
        drop_out: float = 0.2,
        lr: float = 0.01,
        wd: float = 1e-6,
        qlow: Optional[float] = None,
        qhigh: Optional[float] = None,
        use_arrangement: bool = False,
        save_path: Optional[str] = None,
    ):
        self.loss_type = loss_type
        self.optimizer_class = optimizer_class
        self.loss_func = loss_func
        self.test_ratio = test_ratio
        self.CV = CV
        self.all_quantiles = all_quantiles
        self.device = torch.device(device)
        self.nfolds = nfolds
        self.best_epoch = None
        self.save_path = save_path
        self.random_state = random_state
        self.in_shape = in_shape
        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.lr = lr
        self.wd = wd
        self.qlow = qlow
        self.qhigh = qhigh
        self.use_arrangement = use_arrangement
        self.full_loss_history = []
        self.loss_history = []
        self.test_loss_history = []
        self.foldid = None
        self.x = None
        self.y = None

        # Determine output shape
        if loss_type == "mse":
            self.out_shape = 1
        elif loss_type == "qc":
            if all_quantiles is None:
                self.all_quantiles = [qlow, qhigh]
            else:
                self.all_quantiles = all_quantiles
            self.out_shape = len(self.all_quantiles)
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        torch.manual_seed(random_state)
        self.model = network_factory(
            in_shape=in_shape, out_shape=self.out_shape,
            hidden_size=hidden_size, drop_out=drop_out,
        )

        n_folds_actual = nfolds if CV else 1
        self.model_cv = []
        for i in range(n_folds_actual):
            torch.manual_seed(random_state)
            self.model_cv.append(network_factory(
                in_shape=in_shape, out_shape=self.out_shape,
                hidden_size=hidden_size, drop_out=drop_out,
            ))
            self.loss_history.append([np.nan])
            self.test_loss_history.append([np.nan])

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int,
            batch_size: int, print_out: int = 10, verbose: bool = True):
        """Fit the model with optional CV."""
        self.x = x
        self.y = y
        n = x.shape[0]
        best_epoch = epochs

        # Prepare CV splits
        if not self.CV:
            if self.foldid is None:
                data = train_test_split(x, y, self.test_ratio, self.random_state)
                xtrain_list = [data["xtrain"]]
                ytrain_list = [data["ytrain"]]
                xtest = data["xtest"]
                ytest = data["ytest"]
            else:
                xtrain_list, ytrain_list = [], []
                for fid in self.foldid:
                    xtrain_list.append(x[fid])
                    ytrain_list.append(y[fid])
                xtest, ytest = None, None
        else:
            if self.foldid is None:
                data = cv_split(x, y, self.nfolds, self.random_state)
                self.foldid = data["foldid"]
                xtrain_list = data["xtrain"]
                ytrain_list = data["ytrain"]
                xtest_list = data["xtest"]
                ytest_list = data["ytest"]
            else:
                xtrain_list, ytrain_list = [], []
                xtest_list, ytest_list = [], []
                for fid in self.foldid:
                    mask = np.ones(n, dtype=bool)
                    mask[fid] = False
                    xtrain_list.append(x[mask])
                    ytrain_list.append(y[mask])
                    xtest_list.append(x[fid])
                    ytest_list.append(y[fid])

        for k in range(self.nfolds if self.CV else 1):
            if verbose:
                print(f"###CV_fold: {k + 1}")

            best_test_loss = float('inf')
            best_cnt = float('inf')
            cnt = 0

            if not self.CV:
                xtr = xtrain_list[k]
                ytr = ytrain_list[k]
                if xtest is not None:
                    xte_t = torch.tensor(xtest, dtype=torch.float32)
                    yte_t = torch.tensor(ytest, dtype=torch.float32)
            else:
                xtr = xtrain_list[k]
                ytr = ytrain_list[k]
                xte_t = torch.tensor(xtest_list[k], dtype=torch.float32)
                yte_t = torch.tensor(ytest_list[k], dtype=torch.float32)

            for e in range(1, epochs + 1):
                ret = epoch_train(
                    self.model_cv[k], self.loss_func,
                    xtr, ytr, batch_size,
                    self.optimizer_class, self.lr, self.wd,
                    self.device,
                )
                cnt = ret["cnt"]
                self.loss_history[k].append(ret["epoch_loss"])
                self.model_cv[k] = ret["model"]

                # Test evaluation
                self.model_cv[k].eval()
                with torch.no_grad():
                    preds = self.model_cv[k](xte_t)
                    test_loss = self.loss_func(preds, yte_t).item()
                self.test_loss_history[k].append(test_loss)

                if test_loss <= best_test_loss:
                    best_test_loss = test_loss
                    best_epoch = e
                    best_cnt = cnt

                if e % print_out == 0 and verbose:
                    print(f"  Split: Epoch {e}, Train {ret['epoch_loss']:.4f}, "
                          f"Test {test_loss:.4f}, Best Epoch {best_epoch}, "
                          f"Best Loss {best_test_loss:.4f}")

            # Drop initial NaN
            self.test_loss_history[k] = self.test_loss_history[k][1:]
            self.loss_history[k] = self.loss_history[k][1:]

        # Aggregate CV histories
        if self.CV:
            loss_hist = np.array(self.loss_history).T
            test_hist = np.array(self.test_loss_history).T
            self.test_loss_history_agg = test_hist.mean(axis=1)
            self.loss_history_agg = loss_hist.mean(axis=1)
        else:
            self.test_loss_history_agg = np.array(self.test_loss_history[0])
            self.loss_history_agg = np.array(self.loss_history[0])

        self.best_epoch = int(np.argmin(self.test_loss_history_agg)) + 1

        # Refit on all data using best_epoch steps
        for e in range(1, self.best_epoch + 1):
            ret = epoch_train(
                self.model, self.loss_func,
                x, y, batch_size,
                self.optimizer_class, self.lr, self.wd,
                self.device,
            )
            self.model = ret["model"]
            self.full_loss_history.append(ret["epoch_loss"])
            if e % print_out == 0 and verbose:
                print(f"Full: Epoch {e}, Loss {ret['epoch_loss']:.4f}")

    def cv_evaluate(self, num_classes: int, requires_grad: bool = True) -> dict:
        """Cross-validated evaluation with Jacobians."""
        n = self.x.shape[0]
        yhat = np.full((n, num_classes), np.nan)
        jacobians = None
        if requires_grad:
            jacobians = np.full((n, num_classes, self.x.shape[1]), np.nan)

        for k in range(self.nfolds):
            fold_idx = self.foldid[k]
            x_fold = self.x[fold_idx]

            x_tensor = torch.tensor(x_fold, dtype=torch.float32,
                                    requires_grad=requires_grad,
                                    device=self.device)

            self.model_cv[k].eval()
            y1hat = self.model_cv[k](x_tensor)
            yhat[fold_idx, :] = y1hat.detach().cpu().numpy()

            if requires_grad and x_tensor.grad is not None:
                x_tensor.grad.zero_()
            if requires_grad:
                x_jac = compute_jacobian(x_tensor, y1hat)
                jacobians[fold_idx] = x_jac.detach().cpu().numpy()

        return {"yhat": yhat, "jacobians": jacobians}

    def predict(self, x: np.ndarray, requires_grad: bool = True) -> dict:
        """Predict with optional Jacobian computation."""
        x_tensor = torch.tensor(x, dtype=torch.float32,
                                requires_grad=requires_grad,
                                device=self.device)
        self.model.eval()
        yhat = self.model(x_tensor)

        jacobians = None
        if requires_grad:
            jacobians = compute_jacobian(x_tensor, yhat)
            jacobians = jacobians.detach().cpu().numpy()

        return {
            "yhat": yhat.detach().cpu().numpy(),
            "jacobians": jacobians,
        }
