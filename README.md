# Lebedev-cubic
Computing roots of cubic equation explicitly. 

V. I. Lebedev: On formulae for roots of cubic equation
! Sov. J. Numer. Anal. Math. Modelling, Vol.6, No.4, pp. 315-324 (1991)
! (https://www.degruyterbrill.com/document/doi/10.1515/rnam.1991.6.4.315/html)

## Python version (`tc.py`)

A Python port of `tc.f90`, `cubic_roots(A, B, C, D)`, no dependencies beyond
the standard library. Two additions on top of a literal port, found necessary
by testing against a companion-matrix eigenvalue solver (numpy's
`polyroots`) across hundreds of thousands of randomized cubics:

- **Deflation for clustered roots.** When two of the three roots are close
  together, the raw formula can lose accuracy right at the boundary between
  the "three real roots" and "one real + complex pair" cases, a
  floating-point classification problem documented for this whole family of
  formula by W. Kahan (["To Solve a Real Cubic
  Equation"](https://people.eecs.berkeley.edu/~wkahan/Math128/Cubic.pdf),
  1986). Fix: trust whichever root is farthest from the other two (that one
  stays accurate), divide it out of the cubic (Kahan's own deflation
  formulas), and solve the resulting quadratic directly.
- **Kahan-style handling of widely-spread coefficient magnitudes**, using
  his two named degenerate-coefficient patterns plus a general
  variable-rescaling fallback (Kahan sec. 8).
- **Residual/plausibility checks** at the points most likely to silently
  fail: the degenerate-branch anchor root, and the final result before
  it's accepted. Each one is scoped narrowly (checked against the biggest
  or least-precision-starved value available) specifically to avoid the
  false rejections a broad "is this root close to zero" check runs into
  when the true answer legitimately is at or near zero.

Result: typically both faster and more accurate than the eigenvalue
approach, including in the clustered-root case deflation targets, with the
same "raise rather than silently return a wrong answer" guarantee for
coefficients too extreme for any float64 method to resolve.

Validated against a systematic sweep that deliberately pushes pairs of
coefficients to extreme, uncorrelated magnitudes (up to ~300 orders of
magnitude apart, 6 coefficient pairs x thousands of trials each): 0
remaining cases where `cubic_roots` is less accurate than numpy's
eigenvalue-based solver, and 0 cases of a non-finite result returned
without raising, across a separate 200k-trial stress test spanning the
same magnitude range.

Tests: `pytest test_tc.py` (requires `mpmath` and `numpy` for ground-truth
comparisons only; `tc.py` itself needs neither).