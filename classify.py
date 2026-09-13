"""Compare crop representations using the same small 3D CNN.

Archives supply labels, subject IDs and optional train/test splits. Validation
subjects select the epoch; --val-subjects 0 reproduces final-epoch evaluation.

Example: python classify.py --crops data/crops.npz --epochs 40"""

import argparse

import numpy as np
import torch
import torch.nn as nn

CONDITIONS = ["full", "persistence", "persdens", "centre", "random", "recentred", "persist2x"]


class Small3D(nn.Module):
    """A deliberately small network, so the comparison is about the input."""

    def __init__(self, n_classes=11, n_bins=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(2, 16, (3, 3, 3), padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 32, (3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),
            nn.Conv3d(32, 64, (3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.net(x))


def normalise(a):
    a = np.log1p(a)
    m = a.reshape(len(a), -1).max(axis=1)
    m[m == 0] = 1.0
    return (a / m[:, None, None, None, None]).astype(np.float32)


def run_condition(data, name, train_mask, val_mask, test_mask, epochs, seed, device):
    """Return validation-selected and final-epoch test accuracy.

    When val_mask is test_mask, the first value is the historical maximum-test
    diagnostic, not an unbiased selected-epoch result. Use the final value."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = normalise(data[name])
    x = np.transpose(x, (0, 2, 1, 3, 4))
    y = data["label"].astype(np.int64)

    xtr = torch.from_numpy(x[train_mask]).to(device)
    ytr = torch.from_numpy(y[train_mask]).to(device)
    xva = torch.from_numpy(x[val_mask]).to(device)
    yva = torch.from_numpy(y[val_mask]).to(device)
    xte = torch.from_numpy(x[test_mask]).to(device)
    yte = torch.from_numpy(y[test_mask]).to(device)

    model = Small3D(n_classes=int(y.max()) + 1, n_bins=x.shape[2]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.CrossEntropyLoss()

    best_val, chosen, final = -1.0, 0.0, 0.0
    n = len(xtr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 32):
            idx = perm[i : i + 32]
            opt.zero_grad()
            loss = lossf(model(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            va = (model(xva).argmax(1) == yva).float().mean().item()
            te = (model(xte).argmax(1) == yte).float().mean().item()
        final = te
        if va > best_val:
            best_val, chosen = va, te
    return chosen, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="data/crops.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--val-subjects", type=int, default=4)
    ap.add_argument(
        "--conditions",
        default="",
        help="comma-separated subset to score, for pushing more "
        "seeds through one comparison without re-running all",
    )
    ap.add_argument("--seed-start", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.crops)
    users = data["user"]
    labels = data["label"]

    if "split" in data:
        train_mask = data["split"] == 0
        test_mask = data["split"] == 1
        origin = "official train/test split of the dataset"
    else:
        uniq = np.unique(users)
        test_mask = np.isin(users, uniq[-6:])
        train_mask = ~test_mask
        origin = "held-out subjects carved from the training split"

    # Without validation, retain all training subjects and use final-epoch
    # accuracy. The maximum-test diagnostic is retained for historical comparison.
    train_users = np.unique(users[train_mask])
    if args.val_subjects > 0:
        val_users = train_users[-args.val_subjects :]
        val_mask = train_mask & np.isin(users, val_users)
        train_mask = train_mask & ~val_mask
    else:
        val_mask = test_mask
        print("no validation split: training on every recording, reporting the FINAL epoch\n")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"recordings {len(labels)}, classes {len(np.unique(labels))}, device {device}")
    print(f"evaluation: {origin}")
    print(f"train subjects {sorted(set(users[train_mask].tolist()))}")
    print(f"val   subjects {sorted(set(users[val_mask].tolist()))}  (epoch selection)")
    print(f"test  subjects {sorted(set(users[test_mask].tolist()))}")
    print(f"train {train_mask.sum()}, val {val_mask.sum()}, test {test_mask.sum()}")
    print(f"chance accuracy {1 / len(np.unique(labels)):.3f}\n")

    sel_label = "val-chosen" if args.val_subjects > 0 else "MAX-TEST(old)"
    # Older crop archives predate recentred and persist2x, so only score what
    # the file actually holds rather than failing on the ones it does not.
    wanted = (
        [c.strip() for c in args.conditions.split(",") if c.strip()]
        if args.conditions
        else CONDITIONS
    )
    present = [c for c in wanted if c in data]
    missing = [c for c in wanted if c not in data]
    if missing:
        print(f"not in this archive, skipping: {', '.join(missing)}\n")
    seeds = range(args.seed_start, args.seed_start + args.seeds)
    print(f"seeds {list(seeds)}\n")

    print(f"{'condition':>13}{sel_label:>14}{'spread':>9}{'final-epoch':>13}")
    results, finals = {}, {}
    for name in present:
        pairs = [
            run_condition(data, name, train_mask, val_mask, test_mask, args.epochs, s, device)
            for s in seeds
        ]
        results[name] = [p[0] for p in pairs]
        finals[name] = [p[1] for p in pairs]
        print(
            f"{name:>13}{np.mean(results[name]):>14.3f}{np.std(results[name]):>9.3f}"
            f"{np.mean(finals[name]):>13.3f}",
            flush=True,
        )

    if "centre" in results:
        base = np.mean(results["centre"])
        print("\nrelative to the fixed centre crop:")
        for name in present:
            print(f"  {name:>12}: {np.mean(results[name]) - base:+.3f}")

    n = args.seeds
    print("\ndifferences, in standard errors of the difference:")
    pairs = [
        ("persistence", "full"),
        ("persistence", "persdens"),
        ("persistence", "centre"),
        ("centre", "random"),
    ]
    if "recentred" in results:
        pairs += [
            ("recentred", "full"),
            ("recentred", "persistence"),
            ("persist2x", "persistence"),
            ("persist2x", "full"),
        ]
    for a, b in pairs:
        if a not in results or b not in results:
            continue
        ma, sa = np.mean(results[a]), np.std(results[a])
        mb, sb = np.mean(results[b]), np.std(results[b])
        se = np.sqrt(sa**2 / n + sb**2 / n)
        sig = abs(ma - mb) / se if se > 0 else float("inf")
        print(f"  {a:>11} - {b:<11} {ma - mb:>+7.4f}   {sig:>5.1f} se")


if __name__ == "__main__":
    main()
