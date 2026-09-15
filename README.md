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

