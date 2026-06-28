# Day 3 — ANN in PyTorch (MNIST)

## Goal
Rebuild the manually-implemented 2-layer NumPy network from Day 2 as a proper PyTorch `nn.Module`, and train it on MNIST (flattened input, no convolutions) to validate understanding of PyTorch's training mechanics: `nn.Module`, optimizers, and the autograd-driven training loop.

## What this covers
- Loading MNIST via `torchvision.datasets` + `DataLoader`, with pixel values scaled to `[0, 1]`
- Defining a 2-layer feedforward network (`784 → 512 → ReLU → 10`) using `nn.Module`, with `__init__` (persistent layers/weights) separated from `forward()` (per-call data flow)
- Training loop using the `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` cycle
- `nn.CrossEntropyLoss` for multi-class classification (chosen over MSE since the task is relative class ranking, not independent regression to 0/1 targets)
- `Adam` optimizer
- Evaluation loop using `torch.no_grad()` and `torch.argmax` to compute test accuracy

## Architecture
```
Input (28x28 image)
  → Flatten → 784
  → Linear(784, 512)
  → ReLU
  → Linear(512, 10)
  → raw logits (10 classes)
```

## Training setup
| Setting | Value |
|---|---|
| Loss function | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-3) |
| Batch size | 64 |
| Epochs | 10 |
| Device | CPU |

## Result
**Test accuracy: 97.88%** (target: >97%) ✅

## Key things learned / debugged
- `optimizer.step()` doesn't just "optimize" — it reads gradients already computed by `loss.backward()` and updates weights/biases according to the optimizer's update rule. `zero_grad()` is necessary because gradients accumulate by default across calls.
- `nn.Module`'s `__init__`/`forward()` split exists so layer weights are created **once** and persist across every forward/backward pass — not regenerated (and re-randomized) on every call.
- Cross-entropy fits this task because the output is a relative judgment ("which of these 10 classes scores highest"), not independent regression of each output toward 0 or 1 like MSE would do.

## Files
- `ANN_Using_PyTorch.ipynb` — full notebook (data loading, model, training loop, evaluation)
