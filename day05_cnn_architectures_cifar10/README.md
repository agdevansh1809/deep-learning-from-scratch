# Day 5 — CNN Architectures (CIFAR-10)

## Concepts covered
- Classic CNN architectures: LeNet, VGG
- ResNet & skip connections
- The degradation problem (distinct from vanishing gradients): plain deep CNNs show *higher training error* as depth increases, not just worse generalization. Adding more layers should, in theory, never hurt (extra layers could just learn identity mappings) — but in practice, plain stacked conv layers struggle to learn even a trivial identity mapping via gradient descent.
- Why `output = F(x) + x` fixes this: instead of forcing the network to learn a complex identity mapping from scratch, the skip connection makes identity the *default, free* behavior — the network only has to learn to push `F(x)` toward zero, a much easier target for the optimizer than reconstructing `x` exactly.
- Receptive fields

## What was built
A CNN for CIFAR-10 classification in PyTorch, built and iterated on from a 2-conv-layer baseline up to a 3-conv-layer architecture with BatchNorm, data augmentation, and a learning rate schedule. No use of resnet or any other such models.

**Final architecture:**
- Conv(3→32, k=3) → BatchNorm → ReLU → MaxPool(2,2)
- Conv(32→64, k=3) → BatchNorm → ReLU
- Conv(64→128, k=3) → BatchNorm → ReLU
- Flatten → Linear(128×11×11 → 10)

**Training setup:**
- Data augmentation: `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` on training set only
- Optimizer: SGD, initial lr=0.005, momentum=0.9
- LR schedule: `StepLR(step_size=8, gamma=0.5)`
- Batch size: 128
- Epochs: 25

**Result: 82.56% test accuracy** on CIFAR-10.

## Key bugs hit and fixed along the way
1. **Conv2d in/out channel mismatches** — repeatedly mixed up `in_channels` vs `out_channels` across layers; learned to trace each layer's expected input shape from the previous layer's output, not guess.
2. **Flatten called on the wrong tensor, and before the conv layers ran** — an early `forward()` version flattened the raw image instead of the conv output, and created a brand-new untrained `nn.Linear` inside `forward()` every call instead of registering it in `__init__`.
3. **No activation functions** — an early version had zero ReLUs between conv layers, which would have collapsed the whole stack to a single linear transformation (composing linear ops without nonlinearity stays linear).
4. **Unstable training at lr=0.01** — loss bounced between ~1.8–2.06 instead of converging; diagnosed as the optimizer overshooting, fixed by dropping lr to 0.001 (later replaced by a scheduled lr starting at 0.005).
5. **Overfitting after adding data augmentation + cutting epochs simultaneously** — changed two variables at once (added augmentation, cut epochs 12→9), which masked the fact that augmentation needs *more* epochs to converge, not fewer. Isolating the variables fixed it.

## Accuracy progression (debugging trail)
| Change | Test accuracy |
|---|---|
| Baseline 2-conv CNN, lr=0.01 | 32.67% |
| lr dropped to 0.001 | 63.4% |
| + data augmentation, epochs cut too aggressively (9) | 56.73% (regression — diagnosed as needing more epochs with augmentation) |
| + epochs back up to 15 | 63.07% |
| + BatchNorm | 67.8% |
| + 3rd conv layer (32→64→128) + LR scheduler (StepLR) + 25 epochs | **82.56%** |

## Notes / next steps
- 82.56% is near the realistic ceiling for a plain CNN (no skip connections) on CIFAR-10 (~80–85% range) — pushing further from here is where the field's history shows plain deep CNNs hit the degradation problem, which is exactly what motivates ResNet.
- **Deferred to Day 7 (buffer day):** building a custom residual block from scratch and integrating it into this architecture to push past the plain-CNN ceiling.