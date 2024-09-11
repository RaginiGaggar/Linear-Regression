# Linear Regression with Cost Function and Gradient Descent

## Overview

This project implements linear regression from scratch, including the calculation of the cost function and the application of gradient descent to optimize the model parameters. The dataset used is a real estate dataset containing information about house prices, which is visualized and analyzed through various plots. This repository demonstrates the core concepts of linear regression, cost functions, gradient descent, and how to visualize these aspects effectively.

## Table of Contents

- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Data](#data)
- [Computation](#computation)
  - [Cost Function](#cost-function)
  - [Gradient Descent](#gradient-descent)
- [Visualization](#visualization)

## Project Structure

- `linear_regression.ipynb`: Jupyter notebook containing the code for linear regression, cost function computation, and gradient descent.
- `Real_Estate.csv`: Dataset used for demonstration.
- `README.md`: This file.

## Dependencies

Make sure you have the following Python libraries installed:

- `pandas`
- `matplotlib`
- `numpy`
- `scienceplots`
- `seaborn`
- `plotly`

You can install the dependencies using:

```bash
pip install pandas matplotlib numpy scienceplots seaborn plotly
```

## Data

The dataset used is `Real_Estate.csv`, which contains information on house prices and their respective areas. The columns of interest are:

- `House age`: Age of the house (in years)
- `House price of unit area`: Price per unit area of the house (in dollars)

## Computation

### Cost Function

Two versions of the cost function are implemented:

1. **Iterative Approach**: Calculates the cost using a loop.

    ```python
    def cost_function(x, y, w, b):
        total_cost = 0
        for i in range(n):
            total_cost += (y[i] - (w*x[i] + b))**2
        return total_cost / float(n)
    ```

2. **Vectorized Approach**: Uses NumPy operations to compute the cost more efficiently.

    ```python
    def vectorised_cost_function(x, y, w, b):
        total_cost = np.sum((y - (w*x + b))**2) / (2*n)
        return total_cost
    ```

The cost function is defined as:

$\[ J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} (y_i - (w \cdot x_i + b))^2 \]$

where:
- \( w \) is the weight (slope)
- \( b \) is the bias (intercept)
- \( x_i \) and \( y_i \) are the features and target values
- \( n \) is the number of samples

### Gradient Descent

Gradient descent is used to minimize the cost function by iteratively updating the weight and bias:

- **Update Equations**:
  - $\( w := w - \a\cdot \frac{\partial J}{\partial w} \)$
  - $\( b := b - \a\cdot \frac{\partial J}{\partial b} \)$

where:
- \( a \) is the learning rate

The gradient descent function performs updates based on the gradients of the cost function:

```python
def gradient_descent(x, y, w, b, a, iterations):
    w_history = []
    b_history = []
    cost_history = []

    for i in range(iterations):
        dw = -(2/n) * np.sum((y - (w*x + b)) * x)
        db = -(2/n) * np.sum(y - (w*x + b))

        w_history.append(w)
        b_history.append(b)
        cost_history.append(vectorised_cost_function(x, y, w, b))

        w = w - a * dw
        b = b - a * db

    return w, b, w_history, b_history, cost_history
```

## Visualization

### Cost vs Weight

The cost function is plotted against a range of weight values while keeping the bias constant.

### Cost vs Bias

The cost function is plotted against a range of bias values while keeping the weight constant.

### 3D Cost Surface

A 3D surface plot shows the cost function in relation to both weight and bias, providing a comprehensive view of the optimization landscape.

### Contour Plot

A contour plot visualizes the cost function with contour lines, along with the gradient descent path overlaid to show the optimization trajectory.

---

This README should provide a clear and detailed explanation of your project's code and visualizations. Feel free to adjust or expand upon it as needed!
