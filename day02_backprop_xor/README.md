# Day 2 — Backpropagation & Gradient Descent (from scratch, NumPy)

## What this is
A fully manual implementation of backpropagation and gradient descent for a
2-layer neural network — no autograd, no PyTorch. Every gradient (`dW1`, `db1`,
`dW2`, `db2`) is derived from the chain rule and coded by hand.

The network is trained on **`make_moons`** (300 samples, noise=0.2) — a
non-linearly-separable 2D binary classification dataset from scikit-learn —
with a proper train/test split, not just the toy XOR case.

## Architecture
- Input: 2 features
- Hidden layer: 16 neurons, sigmoid activation
- Output layer: 1 neuron, sigmoid activation
- Loss: binary cross-entropy
- Optimizer: vanilla (batch) gradient descent, manual weight updates

```
x (2) → [W1: 2×16, sigmoid] → a1 (16) → [W2: 16×1, sigmoid] → a2 (1)
```

## Key results
- **Train accuracy:** ~96.7%
- **Test accuracy:** ~96.7% (matches train — confirms the model generalized
  rather than memorized)
- Loss curve: sharp drop in the first ~2 epochs (of the 500-step logging
  interval), then flattens — clean convergence, no divergence or oscillation

## What was implemented
- `sigmoid(z)` — manual sigmoid activation
- `forward(x, w, b)` — single-layer forward pass, returns `(z, a)`
- `compute_loss(p, y)` — binary cross-entropy loss
- `backward(z1, a1, z2, a2, x, y, w2)` — full manual backward pass:
  - `dz2 = a2 - y` (clean BCE + sigmoid combination)
  - `dW2 = a1.T @ dz2`, `db2 = sum(dz2)`
  - `da1 = dz2 @ W2.T` → `dz1 = da1 * a1 * (1 - a1)`
  - `dW1 = x.T @ dz1`, `db1 = sum(dz1)`
- Training loop: 10,000 iterations of full-batch gradient descent, loss and
  train accuracy logged every 500 steps
- Train/test split (80/20) via `sklearn.model_selection.train_test_split`,
  with test-set accuracy computed via a separate forward pass


## What this proves
- The hand-derived backward pass isn't just correct for the 4-point XOR toy
  case — it generalizes to a separate (still synthetic, but non-trivial)
  dataset with proper train/test methodology, train accuracy ≈ test
  accuracy confirming no overfitting.
- Concrete derivation of why `dL/dz2 = a2 - y`: the BCE loss gradient
  `dL/da2` and the sigmoid derivative `da2/dz2 = a2(1-a2)` combine and cancel
  exactly to this simple form — verified algebraically, not just taken on
  faith from a formula sheet.
