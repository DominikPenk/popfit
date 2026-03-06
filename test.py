import logging

import torch

import popfit

logging.basicConfig(
    level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
)


# 1. Define the Model
class QuadraticModel(popfit.Model):
    def __init__(self):
        super().__init__()
        # Register variables with bounds
        self.a = popfit.Variable(bounds=(-5.0, 5.0))
        self.b = popfit.Variable(bounds=(-10.0, 10.0))

    def forward(self, x):
        # the .value property automatically returns:
        # - the population during training (shape: (N,))
        # - the global optimum during evaluation (shape: ())
        return self.a.value.unsqueeze(-1) * (x**2) + self.b.value.unsqueeze(-1)


# 2. Generate Synthetic Data
torch.manual_seed(42)
x_data = torch.linspace(-2, 2, 20)
y_true = 2.5 * (x_data**2) + 1.2
y_noisy = y_true + torch.randn_like(y_true) * 0.1

model = QuadraticModel()

# 3. Setup Optimizer
# population_size determines how many candidates are evaluated per step
optimizer = popfit.optim.PSO(model, population_size=32)


# 4. Optimization Loop
with optimizer:
    for generation in range(100):
        # model(x_data) returns an Expression
        # .value evaluates it across the current population
        # resulting shape: (population_size, num_data_points)
        predictions = model(x_data)

        # Calculate MSE loss per individual
        # losses shape: (population_size,)
        losses = ((predictions - y_noisy) ** 2).mean(dim=1)

        # Update the population
        current_best_loss = optimizer.step(losses)

        if generation % 20 == 0:
            print(f"Gen {generation}: Best Loss = {current_best_loss:.2e}")

# 5. Retrieve Results
print(f"Optimized a: {model.get_variable('a').optimal.item():.3f}")
print(f"Optimized b: {model.get_variable('b').optimal.item():.3f}")
