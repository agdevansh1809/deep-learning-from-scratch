# Day 1 — Perceptron from Scratch (NumPy)

## What this is
A from-scratch implementation of a perceptron (no ML frameworks) to classify a synthetic 2D dataset of two linearly separable Gaussian blobs. Two training methods were implemented and compared:

1. **Classic Perceptron Rule** (Rosenblatt) — per-sample update, using a hard step activation.
2. **Gradient Descent + Sigmoid** — batch update using log loss (binary cross-entropy), using a smooth sigmoid activation.

## What's inside
- `perceptron.ipynb` — full notebook: data generation, forward pass, both training methods, convergence plots, decision boundary plots, accuracy evaluation.

## Key results
- Both models reached **100% training accuracy** (expected, since the dataset is linearly separable by construction).
- **Convergence behavior differs:**
  - Perceptron: error count is non-monotonic (rises before falling) since updates only fire on misclassification, with no smooth objective.
  - Gradient descent: loss decreases smoothly from `ln(2) ≈ 0.693` (the expected starting loss for a maximally uncertain binary classifier) down to ~0.057.
- **Boundaries differ:** both lines correctly separate the classes but settle in different positions — perceptron stops at the *first* valid separator it finds, while gradient descent keeps adjusting to minimize loss, settling closer to the center of the gap between classes.

## Notable debugging note
On one run, the perceptron's `w[1]` converged to a value very close to zero. Since the boundary-plotting formula (`x2 = -(w0*x1 + b)/w1`) divides by `w[1]`, this produced an extreme, near-vertical line. Root cause: a near-zero `w[1]` means the learned boundary is geometrically closer to a vertical line (`x1 = const`) than to a sloped one — the plotting formula just isn't built to express that case. (Known edge case, not yet handled)

## Concepts covered
Perceptron mechanics, weights & bias, step vs. sigmoid activation, vectorized forward pass, log loss, gradient derivation/update, batch gradient descent, convergence behavior comparison.