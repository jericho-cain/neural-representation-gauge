import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, d_in: int = 64, d_hidden: int = 64, d_out: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.hidden(x))


class SmallCNN(nn.Module):
    def __init__(self, feature_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.proj = nn.Linear(128, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(start_dim=1)
        return self.proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.hidden(x))


@dataclass
class DigitsData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


@dataclass
class CifarLoaders:
    train_loader: DataLoader
    test_loader: DataLoader


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_digits_data(test_size: float = 0.25, seed: int = 42) -> DigitsData:
    digits = load_digits()
    x = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return DigitsData(
        x_train=torch.from_numpy(x_train),
        y_train=torch.from_numpy(y_train),
        x_test=torch.from_numpy(x_test),
        y_test=torch.from_numpy(y_test),
    )


def train_or_load_model(
    checkpoint_path: Path,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[MLP, DigitsData]:
    set_seed(seed)
    data = load_digits_data(seed=seed)

    model = MLP()
    model.to(device)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, data

    train_ds = TensorDataset(data.x_train, data.y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model, data


def _build_cifar10_loaders(
    data_root: Path,
    batch_size: int = 128,
    num_workers: int = 2,
) -> CifarLoaders:
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return CifarLoaders(train_loader=train_loader, test_loader=test_loader)


def train_or_load_cifar10_model(
    checkpoint_path: Path,
    data_root: Path,
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[SmallCNN, CifarLoaders]:
    set_seed(seed)
    loaders = _build_cifar10_loaders(data_root=data_root, batch_size=batch_size)

    model = SmallCNN()
    model.to(device)

    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, loaders

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loaders.train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    model.eval()
    return model, loaders


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    logits = model(x)
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def representations(model: MLP, x: torch.Tensor) -> torch.Tensor:
    return model.hidden(x)


def pairwise_cosine(h: torch.Tensor) -> np.ndarray:
    h_np = h.detach().cpu().numpy()
    norms = np.linalg.norm(h_np, axis=1, keepdims=True)
    h_norm = h_np / np.clip(norms, 1e-12, None)
    return h_norm @ h_norm.T


def make_invertible_gauge(d: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q1, _ = torch.linalg.qr(a)
    b = torch.randn(d, d, generator=g)
    q2, _ = torch.linalg.qr(b)

    # Non-orthogonal scaling ensures cosine geometry changes.
    scales = torch.logspace(-0.5, 0.5, d)
    return q1 @ torch.diag(scales) @ q2


def make_conditioned_gauge(d: int, kappa: float, seed: int = 42) -> torch.Tensor:
    if kappa < 1.0:
        raise ValueError("kappa must be >= 1.0")

    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q1, _ = torch.linalg.qr(a)
    b = torch.randn(d, d, generator=g)
    q2, _ = torch.linalg.qr(b)

    # Log-spaced singular values yield an approximately controlled condition number.
    if d == 1:
        scales = torch.ones(1)
    else:
        scales = torch.logspace(np.log10(kappa), 0.0, d)
    return q1 @ torch.diag(scales) @ q2


def make_orthogonal_gauge(d: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q, _ = torch.linalg.qr(a)
    return q


def whiten(h: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cov = (h.T @ h) / h.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, eps, None))) @ evecs.T
    h_white = h @ inv_sqrt
    cov_white = (h_white.T @ h_white) / h_white.shape[0]
    return h_white, cov, cov_white


@torch.no_grad()
def collect_hidden_logits_labels(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = []
    logits_all = []
    labels = []

    model.eval()
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        feats = model.hidden(xb)
        features.append(feats.cpu())
        logits_all.append(logits.cpu())
        labels.append(yb.cpu())

    return torch.cat(features, dim=0), torch.cat(logits_all, dim=0), torch.cat(labels, dim=0)


def cosine_neighbors(h: np.ndarray, k: int) -> np.ndarray:
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    h_norm = h / np.clip(norms, 1e-12, None)
    sim = h_norm @ h_norm.T
    np.fill_diagonal(sim, -np.inf)
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    row = np.arange(idx.shape[0])[:, None]
    order = np.argsort(-sim[row, idx], axis=1)
    return idx[row, order]


def mean_jaccard_at_k(h_before: np.ndarray, h_after: np.ndarray, k: int) -> float:
    n_a = cosine_neighbors(h_before, k)
    n_b = cosine_neighbors(h_after, k)
    overlaps = []
    for i in range(n_a.shape[0]):
        a = set(n_a[i].tolist())
        b = set(n_b[i].tolist())
        overlaps.append(len(a & b) / len(a | b))
    return float(np.mean(overlaps))


def mean_jaccard_curve(h_before: np.ndarray, h_after: np.ndarray, k_values: Iterable[int]) -> dict[int, float]:
    return {int(k): mean_jaccard_at_k(h_before, h_after, int(k)) for k in k_values}
