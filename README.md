# PopFit

**PopFit** is a PyTorch-based library for **population-based**, **multistart**, and **gradient-free optimization**.  
It is designed for non-convex, noisy, or derivative-free problems by maintaining a population of candidate solutions and integrating evolutionary strategies directly into the PyTorch ecosystem.

PopFit lets you optimize models using methods like **CMA-ES**, **multistart gradient descent**, and other population-based optimizers—without giving up PyTorch’s composability, devices, or tensor semantics.

---

## Motivation

Standard gradient-based optimization often struggles on **non-convex loss landscapes** with many local minima—especially when the number of parameters is relatively small or gradients are unreliable.

Population-based methods address this by exploring **multiple regions of the search space in parallel**. Multistart strategies further reduce sensitivity to initialization.

PopFit brings these ideas together in a clean, PyTorch-native design:

- Variables hold **entire populations**, not just single values
- Optimizers operate on **loss vectors**, not scalars
- Bounds, parametrizations, and population handling are **first-class concepts**

---


## Minimal Example

This example minimizes `(x - 2)^2` using a population-based optimizer.

```python
import torch
import popfit

class Model(popfit.Model):
    def __init__(self):
        super().__init__()
        self.x = popfit.Variable(bounds=(-5.0, 5.0))

    def forward(self):
        # Minimize (x - 2)^2
        return (self.x.value - 2.0) ** 2

model = Model()
optimizer = popfit.optim.BlockCMAES(model, population_size=16)

with optimizer:
    for _ in range(50):
        losses = model()          # shape: (population_size,)
        optimizer.step(losses)

print("Optimal x:", model.x.optimal.item())
```

---


## Key Features

- **PyTorch-native design**  
  `Variable` and `Model` inherit from `nn.Module` and work seamlessly with tensors, devices, and autograd.

- **Population-based optimizers**  
  Evolutionary strategies (e.g. CMA-ES) and multistart gradient-based methods with a familiar optimizer interface.

- **Symbolic expressions**  
  Build complex objective functions using an `Expression` system that automatically handles population dimensions.

- **Bound constraints**  
  Native support for variable bounds and bounded parametrizations (e.g. sigmoid-bounded variables).

- **Variable parametrizations**  
  Transform optimization spaces to improve numerical stability and convergence.

---

## Installation

Clone the repository and install from source:

```bash
git clone https://github.com/username/popfit.git
cd popfit
pip install -e .
```

> ⚠️ This repository is under active development. API might change.

## Core Concepts

### Variables

A `Variable` represents a parameter to be optimized.  
Unlike `nn.Parameter`, it maintains:

- A **population** of candidate values
- A **global best** value found so far
- Optional bounds and parametrizations

```python
a = popfit.Variable(bounds=(-5.0, 5.0))
```

---

### Models

The `Model` class is a `nn.Module` with additional methods to iterate over variables and expressions in the module. 

```python
class MyModel(popfit.Model):
    ...
```

---

### Optimizers

PopFit optimizers mirror `torch.optim` conceptually, but operate on **populations**.

Key differences:

- `step(losses)` expects a **vector of losses** (one per individual)
- Optimizers may need setup and teardown → use them as **context managers**

---

## Example: Polynomial Fitting with CMA-ES

This example fits a quadratic function `y = ax^2 + b` to noisy data using the **BlockCMAES** optimizer.

---

### 1. Define the Model

```python
import torch
import popfit


class QuadraticModel(popfit.Model):
    def __init__(self):
        super().__init__()
        self.a = popfit.Variable(bounds=(-5.0, 5.0))
        self.b = popfit.Variable(bounds=(-10.0, 10.0))

    def forward(self, x):
        # During training:
        #   self.a.value -> (population_size,)
        # During evaluation:
        #   self.a.value -> ()
        return self.a.value.unsqueeze(-1) * (x ** 2) + self.b.value.unsqueeze(-1)
```

---

### 2. Generate Synthetic Data

```python
torch.manual_seed(42)

x_data = torch.linspace(-2, 2, 20)
y_true = 2.5 * (x_data ** 2) + 1.2
y_noisy = y_true + torch.randn_like(y_true) * 0.1
```

---

### 3. Create Model and Optimizer

```python
model = QuadraticModel()

optimizer = popfit.optim.BlockCMAES(
    model,
    population_size=32,
    sigma=0.5,
)
```

---

### 4. Optimization Loop

```python
with optimizer:
    for generation in range(100):
        predictions = model(x_data)

        # Mean squared error per individual
        losses = ((predictions - y_noisy) ** 2).mean(dim=1)

        best_loss = optimizer.step(losses)

        if generation % 20 == 0:
            print(f"Gen {generation}: Best Loss = {best_loss:.2e}")
```

---

### 5. Retrieve Optimized Parameters

```python
print(f"Optimized a: {model.get_variable('a').optimal.item():.3f}")
print(f"Optimized b: {model.get_variable('b').optimal.item():.3f}")
```

---

## Status & Roadmap

PopFit is experimental but functional. Planned improvements include:

- Additional population-based optimizers
- Better diagnostics and logging
