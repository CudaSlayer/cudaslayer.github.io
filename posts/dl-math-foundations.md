> A concise tour of the core math behind deep learning: vectors and linear maps, losses and likelihoods, gradients and the chain rule, backpropagation, and optimization.

## 1) Data as Vectors and Tensors

- A single input is a vector $x \in \mathbb{R}^d$.
- A batch of $n$ inputs is a matrix $X \in \mathbb{R}^{n\times d}$.
- Images, audio, and sequences are often higher-rank tensors (3D or 4D arrays) but the same linear algebra applies.

Useful norms:

- Euclidean norm: $\lVert x \rVert_2 = \sqrt{\sum_i x_i^2}$
- Mean squared error between $x$ and $y$: $\operatorname{MSE}(x,y)=\tfrac{1}{d}\sum_i (x_i-y_i)^2$

## 2) Linear Models and Affine Maps

The simplest model is linear: $\hat{y} = w^\top x$ (scalar output). Adding a bias and vector output gives an affine map:

$$
\hat{y} = W x + b,\qquad W\in \mathbb{R}^{k\times d},\; b\in \mathbb{R}^{k}.
$$

Stacking affine maps with nonlinearities gives a neural network:

$$
\begin{aligned}
 h^{(1)} &= \sigma\!\left(W^{(1)} x + b^{(1)}\right),\\
 h^{(2)} &= \sigma\!\left(W^{(2)} h^{(1)} + b^{(2)}\right),\\
 \hat{y} &= W^{(3)} h^{(2)} + b^{(3)}.
\end{aligned}
$$

A tiny feedforward sketch:

![Tiny MLP](assets/diagrams/dl-basics.svg)

- Circles are neurons (numbers). Lines are weighted connections.
- The function $\sigma$ is an elementwise nonlinearity (e.g., ReLU $\max(0,z)$ or $\tanh$).

## 3) Losses and Likelihoods

We turn predictions $\hat{y}$ into a scalar objective $\mathcal{L}$:

- Regression (targets $y\in\mathbb{R}$): squared error
  $$
  \mathcal{L}(\hat{y}, y) = \tfrac{1}{2}\,(\hat{y}-y)^2.
  $$

- Binary classification (target $y\in\{0,1\}$): logistic regression
  $$
  p(y=1\mid x) = \sigma(z),\; z=w^\top x + b,\quad \sigma(z)=\frac{1}{1+e^{-z}}.
  $$
  Negative log-likelihood (cross-entropy):
  $$
  \mathcal{L}(\hat{p},y)= -\,y\,\log \hat{p} - (1-y)\,\log(1-\hat{p}).
  $$

- Multiclass (one-hot $y\in\{0,1\}^K$): softmax
  $$
  \hat{p_k} = \frac{e^{z_k}}{\sum_j e^{z_j}},\qquad \mathcal{L}(\hat{p},y)= -\sum_{k=1}^K y_k\,\log \hat{p_k}.
  $$

Regularization (keeps weights small): $\lambda\,\lVert W\rVert_F^2$ is often added to the total loss.

## 4) Gradients and the Chain Rule

Training means minimizing $\mathcal{L}(\theta)$ over parameters $\theta$ (all weights and biases). The gradient is the vector of partial derivatives:

$$
\nabla_\theta \mathcal{L}(\theta) = \left[ \frac{\partial \mathcal{L}}{\partial \theta_1},\dots,\frac{\partial \mathcal{L}}{\partial \theta_m} \right].
$$

The chain rule composes derivatives. For $z = f(g(x))$,
$$
\frac{d z}{dx} = \frac{d f}{d g}\,\frac{d g}{d x}.
$$
In networks, this unrolls layer by layer, right-to-left.

## 5) Backpropagation (by Hand for One Layer)

Take a single sample, squared-error loss, and one affine layer $\hat{y}=W x + b$:

$$
\mathcal{L}= \tfrac{1}{2}\lVert \hat{y} - y \rVert_2^2 = \tfrac{1}{2}\lVert W x + b - y \rVert_2^2.
$$

Let $e = \hat{y} - y$. Then using matrix calculus,

- Gradient w.r.t. outputs: $\tfrac{\partial \mathcal{L}}{\partial \hat{y}} = e$.
- By chain rule:
  $$
  \frac{\partial \mathcal{L}}{\partial W} = e\,x^\top,\qquad \frac{\partial \mathcal{L}}{\partial b} = e,\qquad \frac{\partial \mathcal{L}}{\partial x} = W^\top e.
  $$
These are the building blocks used repeatedly across layers.

With a nonlinearity $h=\sigma(z)$, you multiply by $\sigma'(z)$ at that node (Hadamard/elementwise product).

## 6) Gradient Descent and SGD

- Full-batch gradient descent (learning rate $\eta$):
  $$
  \theta \leftarrow \theta - \eta\,\nabla_\theta\, \mathcal{L}(\theta).
  $$
- Stochastic gradient descent (SGD): use small batches for noisy, faster updates.
- Optimizers like Momentum, Adam, RMSProp rescale and smooth gradients to speed up convergence.

A typical training loop:

```python
for step, (xb, yb) in enumerate(loader):
    # forward pass (compute predictions)
    yhat = model(xb)
    # scalar loss over the batch
    loss = loss_fn(yhat, yb)
    # backward pass (compute gradients)
    loss.backward()
    # update parameters with an optimizer (e.g., Adam)
    opt.step()
    opt.zero_grad()
```

## 7) Capacity, Generalization, and Regularization

- Model capacity grows with parameters and depth; too much capacity can overfit.
- Regularization: weight decay (L2), dropout, data augmentation, early stopping.
- Validation set monitors generalization; test set is for final performance.

## 8) Where to Go Next

- Add convolution and attention layers to the basic blocks above.
- Learn about initialization (He/Xavier), normalization (Batch/LayerNorm), and residual connections.
- Study optimization scales (learning-rate schedules, warmup, cosine decay).

Key idea: deep learning chains simple, differentiable blocks. With a scalar loss and the chain rule, gradients flow backward to update every parameter.
