> Building a minimal, fast blog with just enough features
> to write about deep learning, maths, and physics.

Welcome! This space will host short, focused notes, derivations,
and implementation details. The stack is intentionally small:

- React via CDN
- Markdown for writing
- KaTeX for equations

No bundlers, no heavy frameworks, just static files. This means it
should be fast on GitHub Pages and trivial to maintain.

Some code:

```python
def squared_error(y, y_hat):
    return ((y - y_hat) ** 2).mean()
```

And a tiny inline equation like $e^{i\pi} + 1 = 0$.

Cheers — on to the real content.
