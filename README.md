<div align="center">

# Neuro

**A single-header C++23 neural network library with AVX2/SSE2 acceleration**

[![C++23](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![Header only](https://img.shields.io/badge/header-only-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

</div>

---

Drop `Neuro.h` into your project. That's it — no build system, no dependencies, no configuration.

```cpp
#include "Neuro.h"

auto net = std::make_unique<Static_neuro<784, 256, 128, 10>>(
    activations::ReLU, 0.01, activations::Linear);

net->init(neuro_init::kaiming, 42);
net->train(input, target);
auto out = net->predict(input);
```

---

## Features

- **Header-only** — one `#include` and you're done
- **Two flavors** — compile-time topology (`Static_neuro`) or runtime topology (`Neuro`)
- **SIMD acceleration** — AVX2 and SSE2 dot-product and weight-update kernels, selected automatically
- **13 built-in activations** — ReLU, GELU, Swish, Mish, Tanh, Sigmoid, SELU, ELU, and more
- **Separate output activation** — no more ReLU killing your output layer
- **Xavier & Kaiming initialization** — or supply your own bounds per layer
- **Binary checkpointing** — `save()` / `load()` in one call
- **Zero runtime dependencies** — standard library only

---

## Quick Start

### 1. Copy the header

```bash
cp Neuro.h your_project/
```

### 2. Compile

```bash
# Maximum performance (recommended)
g++ -std=c++23 -O3 -mavx2 -mfma main.cpp -o train

# SSE2 only
g++ -std=c++23 -O3 -msse2 -msse3 main.cpp -o train

# Scalar (no SIMD)
g++ -std=c++23 -O3 main.cpp -o train
```

### 3. Train

```cpp
#include "Neuro.h"
#include <memory>
#include <array>
#include <algorithm>
#include <random>

int main() {
    // Static topology — allocate on the heap (large object, ~1.6 MB for this shape)
    auto net = std::make_unique<Static_neuro<784, 256, 128, 10>>(
        activations::ReLU,   // hidden layers
        0.01,                // learning rate
        activations::Linear  // output layer — linear is correct for MSE loss
    );

    net->init(neuro_init::kaiming, /*seed=*/42);

    // --- your training loop ---
    std::array<double, 784> input;   // fill with normalized pixel values [0, 1]
    std::array<double, 10>  target;  // one-hot encoded label
    target.fill(0.0);
    target[3] = 1.0;

    net->train(input, target);

    // --- inference ---
    auto out = net->predict(input);
    int cls = std::distance(out.begin(), std::max_element(out.begin(), out.end()));

    // --- persist ---
    net->save("model.bin");
}
```

---

## Two Classes

| | `Static_neuro<In, ...Layers>` | `Neuro` |
|---|---|---|
| Topology | Fixed at **compile time** | Set at **runtime** |
| Storage | `std::array` inside the object | Heap (`AlignedVector`) |
| Stack safe? | ❌ Use `make_unique` | ✅ |
| Output activation | ✅ Separate `output_actv` | ❌ Same as hidden |
| Performance | Highest (loops unrolled) | Slightly lower |
| Typical use | Training pipeline | Dynamic architecture search |

### `Static_neuro` — compile-time topology

```cpp
// 784 → 256 → 128 → 10  (two hidden layers, one output layer)
Static_neuro<784, 256, 128, 10>

// Always put on the heap — the object contains all weights as std::array
auto net = std::make_unique<Static_neuro<784, 256, 128, 10>>(
    activations::ReLU,    // hidden activation
    0.01,                 // learning rate
    activations::Linear   // output activation (defaults to Linear if omitted)
);
```

### `Neuro` — runtime topology

```cpp
// Architecture defined at runtime — safe on the stack
Neuro net(784, {256, 128, 10}, activations::ReLU, 0.01);
net.init(neuro_init::kaiming, 42);
```

---

## Activation Functions

```cpp
activations::Linear      // f(z) = z              — output layer for MSE/regression
activations::ReLU        // f(z) = max(0, z)       — default hidden activation
activations::LeakyReLU   // f(z) = z > 0 ? z : 0.01·z
activations::PReLU       // f(z) = z > 0 ? z : 0.25·z
activations::ELU         // f(z) = z > 0 ? z : 0.01·(eᶻ−1)
activations::SELU        // self-normalizing
activations::GELU        // transformer-style smooth activation
activations::Swish       // f(z) = z·sigmoid(z)
activations::Mish        // f(z) = z·tanh(softplus(z))
activations::Softplus    // f(z) = log(1 + eᶻ)
activations::Sigmoid     // f(z) = 1/(1+e⁻ᶻ)     — binary classification output
activations::Tanh        // f(z) = tanh(z)
activations::ReLU6       // f(z) = min(max(0,z),6) — mobile networks
```

Custom activations are supported — just supply a `activation_func` struct with two function pointers.

---

## Weight Initialization

```cpp
net->init(neuro_init::kaiming, /*seed=*/42);         // Kaiming uniform — best for ReLU
net->init(neuro_init::kaiming, 42, /*a=*/0.01);      // Kaiming with LeakyReLU slope
net->init(neuro_init::xavier,  /*seed=*/42);          // Xavier uniform — best for Tanh/Sigmoid

// Per-layer bounds — Neuro only
dyn_net.init({ {-0.1, 0.1}, {-0.05, 0.05}, {-0.01, 0.01} });
```

Biases are always initialized to zero. Pass `seed = 0` to use `std::random_device`.

---

## MNIST Example Results

Architecture `784 → 256 → 128 → 10`, ReLU hidden, Linear output, Kaiming init.

| Epochs | LR schedule | Test accuracy |
|---|---|---|
| 10 | constant 0.001 (old, ReLU output) | 88.0% |
| 15 | `0.01 / (1 + 0.5·epoch)` (fixed) | ~93–95% |

The main gains come from three fixes applied by this library's design:
- Linear output activation (no more ReLU clamping gradients at the output)
- Higher initial LR with decay
- Larger hidden layers

---

## API Reference

### Core methods (both classes)

```cpp
// Initialize weights
void init(neuro_init::limit_func fn, uint32_t seed = 0, double a = 0.0);

// Forward pass — returns span into internal buffer, valid until next call
[[nodiscard]] auto predict(const auto& input) noexcept;

// One SGD step: forward → backward → weight update
void train(const auto& input, const auto& target) noexcept;

// Persist
bool save(const std::filesystem::path& path) const noexcept;
bool load(const std::filesystem::path& path)       noexcept;

// Hyperparameters
double learning_rate()              const noexcept;
void   set_learning_rate(double lr)       noexcept;
```

### `Static_neuro` compile-time constants

```cpp
Static_neuro<784, 256, 128, 10>::kInputSize;    // 784
Static_neuro<784, 256, 128, 10>::kOutputSize;   // 10
Static_neuro<784, 256, 128, 10>::kNumLayers;    // 3
Static_neuro<784, 256, 128, 10>::kTotalWeights; // total doubles allocated
```

### `Neuro` extras

```cpp
static Neuro from_file(const path& p, const activation_func& actv); // factory
bool   ready()                   const noexcept;
void   set_activation(const activation_func& actv);
```

---

## Design Notes

**Why MSE and not cross-entropy?** The library deliberately keeps the loss function outside its scope — `train()` only performs one SGD step given a target vector. You choose the target encoding. For classification use one-hot targets; for regression use raw scalar targets.

**Why separate output activation?** The original single-activation design applied ReLU to the output layer, which clips negative logits and cripples classification. The library now accepts a second `output_actv` parameter (defaulting to `Linear`) so the hidden and output activations are independently configurable.

**Why SIMD padding?** Weight rows are padded to `NEURO_SIMD_WIDTH` doubles so the AVX2 kernel can always use aligned 256-bit loads (`_mm256_load_pd`) on weight data. Input data uses unaligned loads (`_mm256_loadu_pd`) since its address cannot be guaranteed. The padding bytes are always zero and do not affect the result.

---

## Requirements

- C++23 or later
- GCC ≥ 13 / Clang ≥ 16 / MSVC ≥ 19.34
- No external dependencies

---

## License

MIT — see `LICENSE` for details.
