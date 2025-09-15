> Goal: Understand how matrices multiply, how affine transformations act on points, and how to differentiate matrix expressions — with proper mathematical notation that still feels approachable.

## 1) Vectors, Matrices, and Shapes

- A column vector with 3 entries is written
  $$
  x = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} \in \mathbb{R}^{3\times 1}.
  $$
- A matrix with 2 rows and 3 columns is

  $$
  A = \begin{bmatrix}
    a_{11} & a_{12} & a_{13} \\
    a_{21} & a_{22} & a_{23}
  \end{bmatrix} \in \mathbb{R}^{2\times 3}.
  $$

Rule of thumb for multiplying shapes:
- If $A\in\mathbb{R}^{m\times n}$ and $B\in\mathbb{R}^{n\times p}$, then $AB\in\mathbb{R}^{m\times p}$.
- The “inner” sizes must match (here: $n$). The result size is the pair of the “outer” sizes ($m$ by $p$).

## 2) Matrix Multiplication = Many Dot Products

Given $A\in\mathbb{R}^{m\times n}$ and $B\in\mathbb{R}^{n\times p}$, their product $C=AB$ has entries
$$
C_{ij} = \sum_{k=1}^n A_{ik}\,B_{kj}.
$$
That is, the $i$-th row of $A$ dotted with the $j$-th column of $B$.

Concrete example ($2\times 3$ times $3\times 1$):
$$
A = \begin{bmatrix}
  2 & 0 & -1 \\
  1 & 3 & 4
\end{bmatrix},\quad
x = \begin{bmatrix} 5 \\ 2 \\ 1 \end{bmatrix},\quad
Ax = \begin{bmatrix}
  2\cdot 5 + 0\cdot 2 + (-1)\cdot 1 \\
  1\cdot 5 + 3\cdot 2 + 4\cdot 1
\end{bmatrix}
= \begin{bmatrix} 9 \\ 15 \end{bmatrix}.
$$

Important properties (that we’ll use later):
- Associative: $(AB)C = A(BC)$ when dimensions line up.
- Distributive: $A(B+C) = AB + AC$.
- Generally not commutative: $AB \neq BA$ in most cases.

## 3) Affine Transformations: Linear + Shift

A linear map multiplies by a matrix: $x \mapsto Ax$.
An affine map adds a shift (bias) vector $b$:
$$
\boxed{\;x \mapsto Ax + b\;}
$$
- $A$ rotates/scales/shears the space.
- $b$ shifts the whole space.

Small 2D example with a shear and shift:
$$
A = \begin{bmatrix} 1 & 0.5 \\ 0 & 1 \end{bmatrix},\quad
b = \begin{bmatrix} 1 \\ -1 \end{bmatrix},\quad
x = \begin{bmatrix} 2 \\ 1 \end{bmatrix}
\;\Rightarrow\; Ax+b = \begin{bmatrix} 2+0.5\cdot 1 \\ 1 \end{bmatrix}+\begin{bmatrix} 1 \\ -1 \end{bmatrix}
= \begin{bmatrix} 2.5 \\ 0 \end{bmatrix}.
$$

A picture helps. The square below is sheared by $A$ and then shifted by $b$.

![Affine square](assets/diagrams/affine-square.svg)

Homogeneous coordinates (optional, cool):
- Write points as $\tilde{x}=\begin{bmatrix}x\\1\end{bmatrix}$.
- Then any affine map becomes a single matrix multiply:
  $$
  T = \begin{bmatrix} A & b \\ 0 & 1 \end{bmatrix},\qquad \tilde{y} = T\,\tilde{x}.
  $$
This trick lets us compose transforms by just multiplying the big $T$ matrices.

## 4) Functions You Already Know (as Matrices)

- Linear regression prediction: $\hat{y} = w^\top x$ is a matrix multiply with a row vector.
- Multi-output regression: $\hat{y} = Wx + b$ is an affine map.
- A neural network layer is exactly that: $h=\sigma(Wx+b)$ (add a nonlinearity $\sigma$).

Composition: if $f(x)=A_2(A_1 x + b_1) + b_2$, then
$$
\hat{A} = A_2 A_1,\qquad \hat{b} = A_2 b_1 + b_2,\qquad f(x)=\hat{A}x+\hat{b}.
$$
It’s still affine — composition of affine maps is affine.

## 5) Distances, Angles, and Quadratic Forms

- Length (Euclidean norm): $\lVert x \rVert_2 = \sqrt{x^\top x}$.
- Distance: $\lVert x-y \rVert_2$.
- Quadratic form: $x^\top Q x$. If $Q$ is symmetric positive definite, it acts like a “weighted” squared length.

Example:
$$
\tfrac{1}{2}\lVert Ax-b \rVert_2^2 = \tfrac{1}{2} (Ax-b)^\top (Ax-b)
$$
shows up everywhere (least squares, energy minimization, etc.).

## 6) Matrix Calculus: Differentiating with Respect to Vectors and Matrices

We need derivatives to learn (optimize). The gradient w.r.t. a vector $x$ is
$$
\nabla_x f(x) = \begin{bmatrix} \tfrac{\partial f}{\partial x_1} \\ \vdots \\ \tfrac{\partial f}{\partial x_d} \end{bmatrix},\quad
\text{and the Jacobian } J_f(x) = \frac{\partial f}{\partial x} \in \mathbb{R}^{m\times d}\text{ if } f: \mathbb{R}^d\to\mathbb{R}^m.
$$

Useful identities (matrix calculus “mini-cheat-sheet”):
- $\displaystyle \frac{\partial}{\partial x}(a^\top x) = a$.
- $\displaystyle \frac{\partial}{\partial x}\,\tfrac{1}{2}\lVert x \rVert_2^2 = x$.
- $\displaystyle \frac{\partial}{\partial x}(x^\top A x) = (A + A^\top)\,x$.
- $\displaystyle \nabla_x\, \tfrac{1}{2}\lVert Ax - b \rVert_2^2 = A^\top (Ax - b)$.
- If $L = \tfrac{1}{2}\lVert Wx - y \rVert_2^2$, then
  $$
  \boxed{\;\nabla_W L = (Wx - y)\,x^\top\;}\qquad\text{and}\qquad
  \boxed{\;\nabla_x L = W^\top (Wx - y).\;}
  $$
  These are the backbone of linear-layer backpropagation.

Chain rule (vector form): If $z = f(g(x))$,
$$
\frac{\partial z}{\partial x} = \frac{\partial f}{\partial g}\;\frac{\partial g}{\partial x}.
$$
For scalars you multiply derivatives; for vectors/matrices you multiply Jacobians with the right shapes.

## 7) Worked Example: One Gradient Step

Let
$$
A=\begin{bmatrix}2&0\\1&3\end{bmatrix},\quad b=\begin{bmatrix}1\\-1\end{bmatrix},\quad
x=\begin{bmatrix}1\\2\end{bmatrix},\quad y=\begin{bmatrix}4\\0\end{bmatrix},\quad
L(W)=\tfrac{1}{2}\lVert Wx + b - y \rVert_2^2.
$$
Then the residual is $r=Wx+b-y$. The gradient is $\nabla_W L = r\,x^\top$ and $\nabla_b L = r$.
If $W$ starts at $A$ above, compute $r = Ax+b-y$:
$$
Ax = \begin{bmatrix}2\\7\end{bmatrix},\quad r = \begin{bmatrix}2\\7\end{bmatrix}+\begin{bmatrix}1\\-1\end{bmatrix}-\begin{bmatrix}4\\0\end{bmatrix}=\begin{bmatrix}-1\\6\end{bmatrix}.
$$
Thus
$$
\nabla_W L = \begin{bmatrix}-1\\6\end{bmatrix}\begin{bmatrix}1&2\end{bmatrix}=\begin{bmatrix}-1 & -2 \\ 6 & 12\end{bmatrix},\qquad
\nabla_b L = \begin{bmatrix}-1\\6\end{bmatrix}.
$$
A single gradient-descent step with step size $\eta$ yields
$$
W \leftarrow W - \eta\,\nabla_W L,\qquad b \leftarrow b - \eta\,\nabla_b L.
$$

## 8) Why This Matters in Deep Learning

Every layer does an affine map ($Wx+b$) followed by a nonlinearity $\sigma$. Training tweaks $W$ and $b$ to reduce a loss. Matrix multiplication handles the “many dot products at once” efficiently; matrix calculus tells us how to adjust parameters.

Takeaways:
- Matrix multiply is a compact way to do many dot products.
- Affine maps are linear transforms plus a shift.
- Matrix calculus gives gradients like $A^\top(Ax-b)$ and $(Wx-y)x^\top$ that power learning.

If you want a follow-up, we can add determinants, inverses, and geometric meaning (areas/volumes) next.
