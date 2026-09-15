# Logistic Regression from Scratch

A clean, documented implementation of Binary Logistic Regression built purely in Python using **NumPy**, complete with mathematical derivations.

## Features
- **Zero Heavy Dependencies**: Built using basic linear algebra in NumPy.
- **Detailed Math**: In-code step-by-step chain rule derivations for gradients.
- **Loss Diagnostics**: Logs binary cross-entropy loss across training epochs.

## Math Breakdown
### 1. Hypothesis Formulation
$$z = Xw + b$$
$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

### 2. Loss Function (Binary Cross-Entropy)
$$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

### 3. Gradient Calculation
$$\frac{\partial J}{\partial w} = \frac{1}{m} X^T (\hat{y} - y)$$
$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum (\hat{y} - y)$$

# Logistic Regression using Newton's Method (IRLS)

A clean Python implementation of Binary Logistic Regression optimized via **Newton's Method** (Iterative Reweighted Least Squares).

## Gradient Descent vs. Newton's Method
| Feature | Gradient Descent | Newton's Method |
|---|---|---|
| **Optimization** | 1st Order (Gradients) | 2nd Order (Gradient + Hessian) |
| **Learning Rate ($\alpha$)** | Requires Tuning | Automatically scaled by curvature |
| **Convergence Speed** | 1,000+ iterations | 5–15 iterations |
| **Cost per Step** | $\mathcal{O}(m \cdot n)$ | $\mathcal{O}(m \cdot n^2 + n^3)$ |

## Derivation Summary

### 1. Gradient Vector ($\mathbf{g}$)
$$\nabla J(w) = \frac{1}{m} X^T (\hat{y} - y)$$

### 2. Hessian Matrix ($\mathbf{H}$)
$$\mathbf{H} = \nabla^2 J(w) = \frac{1}{m} X^T \mathbf{S} X$$
where $\mathbf{S}$ is a diagonal matrix with entries $S_{ii} = \hat{y}_i(1 - \hat{y}_i)$.

### 3. Parameter Update
$$w^{(k+1)} = w^{(k)} - \mathbf{H}^{-1} \nabla J(w^{(k)})$$

