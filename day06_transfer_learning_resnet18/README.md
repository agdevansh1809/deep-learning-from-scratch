# Day 6 — Transfer Learning with Pretrained ResNet18

## Objective
Fine-tune a pretrained CNN on a small custom dataset instead of training from scratch, using feature extraction (frozen backbone).

## Dataset
Small cats-vs-dogs subset (Kaggle), split into:
- train: 275 images (cat/dog)
- val: 70 images (cat/dog)

## Approach
- Loaded `resnet18` pretrained on ImageNet (`weights='IMAGENET1K_V1'`)
- Froze all backbone parameters (`requires_grad = False`)
- Replaced the final FC layer (`model.fc`) with a new `nn.Linear(512, 2)` for binary classification — only this layer is trainable
- Optimizer (`SGD`) scoped to `model.fc.parameters()` only
- Loss: `CrossEntropyLoss`
- LR scheduler: `StepLR` (step_size=8, gamma=0.5)
- Trained for 25 epochs, saved best model weights by validation accuracy

## Why feature extraction over full fine-tuning
With only 275 training images, fine-tuning the entire network risks overfitting and can destroy useful pretrained features (catastrophic forgetting). Freezing the backbone and training only the final layer keeps the parameter count small relative to dataset size, which is far more robust on limited data.

## Result
**Best validation accuracy: 97.14%**

Train accuracy plateaued lower (~86-88%) than validation accuracy throughout training — investigated and attributed to two factors: (1) training transforms (`RandomResizedCrop`, `RandomHorizontalFlip`) make training examples harder than the deterministic val transforms (`Resize` + `CenterCrop`), and (2) the val set is small (70 images), so accuracy moves in large discrete steps and isn't a stable signal at this scale.

## Bugs hit and fixed
- `model.toDevice(device)` — not a real method; corrected to `model.to(device)`
- `preds = torch.max(outputs, 1)` — `torch.max` with `dim` returns a `(values, indices)` tuple; needed to unpack as `_, preds = torch.max(outputs, 1)` to get class predictions correctly
- Loop indentation bug — best-model loading and final print statements were nested inside the epoch loop instead of after it, which would have loaded a half-trained checkpoint mid-training

## Key takeaway
Transfer learning isn't primarily about saving compute/time — on small datasets, it's often the *only* viable way to get a usable model, since there isn't enough data to learn good general features from scratch.