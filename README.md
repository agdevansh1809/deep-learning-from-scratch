# 15-Day Deep Learning Roadmap — From Scratch

Structured self-study: ANNs → CNNs → RNNs → Transformers → GANs, building 
core architectures from scratch (NumPy) before moving to PyTorch implementations.

## Progress Log

| Day | Topic | What I Built |
|-----|-------|---------------|
| 1 | Perceptron fundamentals | Perceptron from scratch in NumPy (Rosenblatt rule + gradient descent), 100% acc on toy 2D dataset |
| 2 | Backprop + gradient descent | Manual 2-layer sigmoid net in NumPy, XOR + make_moons (~96.7% acc) |
| 3 | ANNs in PyTorch | Rebuilt Day 2 net in PyTorch, MNIST ANN, 97.88% test accuracy |
| 4 | Convolution mechanics (kernels, stride, padding, parameter sharing) | Implemented conv2d() from scratch in NumPy |
| 5 | CNN Architectures (CIFAR-10) | Built CNN from scratch (Conv→BatchNorm→ReLU ×3, augmentation, LR scheduler) — 82.56% test accuracy, near ceiling for a plain CNN without skip connections|