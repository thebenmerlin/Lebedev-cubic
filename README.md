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

Result: typically both faster and more accurate than the eigenvalue
approach, including in the clustered-root case deflation targets, with the
same "raise rather than silently return a wrong answer" guarantee for
coefficients too extreme for any float64 method to resolve.

Tests: `pytest test_tc.py` (requires `mpmath` and `numpy` for ground-truth
comparisons only; `tc.py` itself needs neither).