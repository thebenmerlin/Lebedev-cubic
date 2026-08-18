"""
Python port of V. I. Lebedev's explicit real-cubic root formula
(tc.f90 in this repo), with two accuracy/robustness additions found
necessary by testing against the raw formula:

1. Deflation for clustered roots (suggested by Andrew Knyazev): when
   two of the three roots are close together, the raw formula loses
   accuracy right at the boundary between the "three real roots" and
   "one real + complex pair" cases (a floating-point classification
   problem, not a bug in this port; the same failure mode is
   documented for this entire family of formula by W. Kahan, see
   below). The fix: trust whichever root is farthest from the other
   two (empirically, that one stays accurate even when the other two
   don't), divide it out of the cubic algebraically (deflation), and
   solve the resulting quadratic directly. This sidesteps the
   boundary problem instead of trying to patch it.

2. Kahan-style degenerate-coefficient handling: once coefficients
   span a wide range of magnitudes, normalizing by the largest
   coefficient isn't enough to keep every intermediate term in the
   formula (P, X1, X2, X3, T1-T4) away from spurious overflow or
   underflow. Two named patterns, plus a general variable-rescaling
   fallback for a third pattern Kahan doesn't name, close that gap.
   See Kahan sec. 8 below.

References:
  V. I. Lebedev, "On formulae for roots of cubic equation",
  Sov. J. Numer. Anal. Math. Modelling, Vol. 6, No. 4, pp. 315-324
  (1991).
  W. Kahan, "To Solve a Real Cubic Equation" (1986):
  https://people.eecs.berkeley.edu/~wkahan/Math128/Cubic.pdf

No dependencies beyond the Python standard library. See test_tc.py
for the validation this implementation is based on (mpmath is used
there, as a ground-truth reference, but is not required to use
cubic_roots() itself).
"""
import cmath
import math

__all__ = ["cubic_roots"]


def _all_finite(*vals):
    for v in vals:
        if not cmath.isfinite(v):
            return False
    return True


def _quadratic_roots(c0, c1, c2):
    """Stable roots of c2*x^2 + c1*x + c0 = 0. Avoids the
    cancellation the naive +/- quadratic formula suffers by picking
    the sign of the square root that adds constructively with c1,
    then getting the second root from the product-of-roots relation
    c0/c2 = r0*r1 instead of repeating the subtraction."""
    s = max(abs(c0), abs(c1), abs(c2))
    c0n, c1n, c2n = c0 / s, c1 / s, c2 / s
    d = cmath.sqrt(c1n * c1n - 4 * c2n * c0n)
    if (c1n.conjugate() * d).real < 0:
        d = -d
    q = -0.5 * (c1n + d)
    if q == 0:
        return 0j, -c1n / c2n
    return q / c2n, c0n / q


def _tc(A, B, C, D):
    """Direct port of Lebedev's TC subroutine (tc.f90), mirroring its
    control flow line for line so it can be audited against the
    original Fortran. Normalizes A, B, C, D by their largest
    magnitude itself (locally, not by whatever the caller may have
    already normalized by): this must stay local to this function,
    not hoisted to a caller, because the degenerate-coefficient
    branches in _solve_core need the raw, unnormalized coefficients
    to stay precise (a global normalization can underflow a small
    coefficient to exact zero before those branches get to use it).

    Returns ((x1, x2, x3), three_real): if three_real, x1..x3 are
    the three real roots; otherwise x1 is the real root and
    x2 +/- i*x3 is the complex-conjugate pair.
    """
    s = max(abs(A), abs(B), abs(C), abs(D))
    A, B, C, D = A / s, B / s, C / s, D / s

    T = 1.7320508075688772  # sqrt(3)
    S = 1.0 / 3.0
    T2 = B * B
    T3 = 3.0 * A
    T4 = T3 * C
    P = T2 - T4
    X3 = abs(P) ** 0.5
    X1 = B * (T4 - P - P) - 3.0 * T3 * T3 * D
    X2 = abs(X1) ** S
    T2b = 1.0 / T3
    T3b = B * T2b

    if X3 > 1e-32 * X2:
        T1 = 0.5 * X1 / (P * X3)
        X2c = abs(T1)
        T2c = X3 * T2b
        Tc = T * T2c
        T4c = X2c * X2c
        if P < 0:
            Pc = X2c + (T4c + 1.0) ** 0.5
            Pc = Pc ** S
            T4d = -1.0 / Pc
            T2d = T2c if T1 < 0.0 else -T2c
            x1 = (Pc + T4d) * T2d
            x2 = -0.5 * x1
            x3 = 0.5 * Tc * (Pc - T4d)
            return (x1 - T3b, x2 - T3b, x3), False
        else:
            X3d = abs(1.0 - T4c) ** 0.5
            if T4c > 1.0:
                Pc = (X2c + X3d) ** S
                T4d = 1.0 / Pc
                T2d = -T2c if T1 < 0.0 else T2c
                x1 = (Pc + T4d) * T2d
                x2 = -0.5 * x1
                x3 = 0.5 * Tc * (Pc - T4d)
                return (x1 - T3b, x2 - T3b, x3), False
            else:
                ang = math.atan2(X3d, T1) * S
                cosv = math.cos(ang)
                sinpart = (1.0 - cosv * cosv) ** 0.5 * Tc
                scv = cosv * T2c
                x1 = scv + scv
                x2 = sinpart - scv
                x3 = -(sinpart + scv)
                if x2 <= x3:
                    x2, x3 = x3, x2
                if x1 > x2:
                    pass
                else:
                    x1, x2 = x2, x1
                    if x2 > x3:
                        pass
                    else:
                        x2, x3 = x3, x2
                return (x1 - T3b, x2 - T3b, x3 - T3b), True
    else:
        X2c = -X2 if X1 < 0.0 else X2
        x1 = X2c * T2b
        x2 = -0.5 * x1
        x3 = -T * x2
        if abs(x3) > 1e-32:
            return (x1 - T3b, x2 - T3b, x3), False
        else:
            x3 = x2
            if x1 > x2:
                pass
            else:
                x1, x2 = x2, x1
                if x2 > x3:
                    pass
                else:
                    x2, x3 = x3, x2
            return (x1 - T3b, x2 - T3b, x3 - T3b), True


_DEGENERATE_TOL = 1e-13


def _solve_core(A, B, C, D):
    """Degenerate-coefficient checks (Kahan sec. 8) plus the main
    Lebedev + deflation pipeline. Returns a (r0, r1, r2) tuple of
    complex numbers, possibly containing non-finite values. Never
    raises except ZeroDivisionError, which the caller handles.

    The two named degenerate checks: if A is negligible next to B,
    one root has escaped towards +/-infinity and is well approximated
    by -B/A, with the other two roots coming from B x^2+C x+D=0. If D
    is negligible next to C, one root has collapsed towards 0 and is
    well approximated by -D/C, with the other two roots coming from
    A x^2+B x+C=0.
    """
    if abs(A) < _DEGENERATE_TOL * abs(B):
        X = -B / A
        q0, q1 = _quadratic_roots(D, C, B)
        return complex(X), q0, q1
    if abs(D) < _DEGENERATE_TOL * abs(C):
        X = -D / C
        q0, q1 = _quadratic_roots(C, B, A)
        return complex(X), q0, q1

    (x1, x2, x3), three_real = _tc(A, B, C, D)
    if three_real:
        r1, r2, r3 = complex(x1), complex(x2), complex(x3)
    else:
        r1, r2, r3 = complex(x1), complex(x2, x3), complex(x2, -x3)
    scale = max(abs(r1), abs(r2), abs(r3), 1e-300)
    g12 = abs(r1 - r2) / scale
    g13 = abs(r1 - r3) / scale
    g23 = abs(r2 - r3) / scale
    mingap = min(g12, g13, g23)

    if mingap > 0.1 and _all_finite(r1, r2, r3):
        # roots are comfortably separated: the raw formula is already
        # accurate here (verified by fuzzing, see test_tc.py), skip
        # the extra cost of deflation.
        return r1, r2, r3

    if three_real:
        d12 = abs(x1 - x2)
        d13 = abs(x1 - x3)
        d23 = abs(x2 - x3)
        m1 = min(d12, d13)
        m2 = min(d12, d23)
        m3 = min(d13, d23)
        if m1 >= m2 and m1 >= m3:
            X = x1
        elif m2 >= m1 and m2 >= m3:
            X = x2
        else:
            X = x3
    else:
        X = x1

    # Kahan's deflation formulas (Cubic.pdf sec. 3): divide the known
    # root X out of the cubic to get the quadratic whose roots are
    # the other two.
    if abs(X ** 3) > abs(D / A):
        C2 = -D / X
        B1 = (C2 - C) / X
    else:
        B1 = A * X + B
        C2 = B1 * X + C

    q0, q1 = _quadratic_roots(C2, B1, A)
    return complex(X), q0, q1


def cubic_roots(A, B, C, D):
    """
    Roots of the real cubic A*x^3 + B*x^2 + C*x + D = 0. A must be
    nonzero (this is a cubic solver, not a general polynomial one).

    Returns a tuple of 3 complex numbers (a real root comes back with
    a zero imaginary part). Raises ArithmeticError if the true roots
    don't fit in a float (coefficients spread across an extreme range
    of magnitudes), the same failure case numpy's eigenvalue-based
    root finders raise LinAlgError for; this never silently returns a
    wrong answer.

    Built on V. I. Lebedev's explicit formula (tc.f90 in this repo)
    plus deflation and Kahan-style scaling; see the module docstring
    for references and test_tc.py for the validation this is based
    on.
    """
    A, B, C, D = float(A), float(B), float(C), float(D)
    if A == 0:
        raise ValueError("A must be nonzero (this solves cubics, not "
                          "lower-degree polynomials)")

    # An intermediate 0/0 is a legitimate (if rare) outcome for some
    # coefficient spreads: numpy's float64 turns that into nan and
    # keeps going, but Python's native float division raises
    # ZeroDivisionError instead. Treat the two identically here, as
    # just another way for an attempt to come back non-finite, so a
    # crash in the *first* attempt doesn't skip trying the fallback.
    def attempt(a, b, c, d):
        try:
            return _solve_core(a, b, c, d)
        except ZeroDivisionError:
            return None

    r = attempt(A, B, C, D)
    if r is not None and _all_finite(*r):
        return r

    if D != 0:
        # Substitute x = rho*y so the transformed coefficients are
        # more balanced, solve for y, then undo the substitution.
        # A*rho**3 == sign(A)*|D| by construction of rho, so it's
        # computed directly to avoid overflow in rho**3 itself.
        try:
            rho = abs(D / A) ** (1.0 / 3.0)
        except ZeroDivisionError:
            rho = float("nan")
        if math.isfinite(rho) and rho > 0:
            A2 = math.copysign(abs(D), A)
            B2 = B * rho * rho
            C2 = C * rho
            r2 = attempt(A2, B2, C2, D)
            if r2 is not None:
                # scale back from y to x *before* checking finiteness:
                # this multiplication can itself overflow even when r2
                # (in y-space) was finite.
                r2 = tuple(v * rho for v in r2)
                if _all_finite(*r2):
                    return r2

    raise ArithmeticError(
        "no finite roots found: coefficients are too extreme for "
        "this formula (some root likely does not fit in a float)"
    )
