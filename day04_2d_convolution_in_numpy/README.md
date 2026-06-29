# Day 4 — Convolution Mechanics

## Topic
Kernels/filters, stride, padding, feature maps, pooling, and parameter sharing — why CNNs outperform fully-connected ANNs on image data.

## What was built
Implemented `conv2d()` from scratch in NumPy — a manual sliding-window convolution operation supporting:
- Arbitrary kernel sizes (including non-square kernels)
- Configurable stride
- Zero-padding (via `np.pad`)

No `nn.Conv2d` used — pure NumPy, to understand the underlying sliding-window math before relying on framework abstractions.

## Key concepts
- **Parameter sharing**: a single filter's weights are reused across every spatial position in the image, based on the assumption that a feature useful in one region is useful everywhere. This is why CNNs need drastically fewer parameters than a fully-connected layer would for the same image.
- **Output size formula**: `(input_dim + 2*padding - kernel_dim) / stride + 1`
- **Padding**: expands the input before convolving so border pixels get the same coverage as interior pixels, preventing edge information loss across stacked conv layers.

## Bugs hit & fixed
- Swapped kernel height/width (`kernel.shape[0]` vs `[1]`) — stayed silently broken on square test kernels, only surfaced when tested with a non-square kernel.
- Used output-grid indices (`i`, `j`) directly as input-slicing indices — only coincidentally correct at `stride=1`; fixed by scaling with `i*stride`, `j*stride`.

## Validation
Tested against hand-calculated output shapes for:
- Square kernel, stride=1, no padding → `(5,5)` on a 7x7 input
- Non-square kernel (2x3) → `(6,3)` on a 7x5 input
- Stride=2 with padding=1 → `(4,4)` on an 8x8 input