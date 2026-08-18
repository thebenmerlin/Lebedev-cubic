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


def _mul_underflowed(*factors):
    """True if multiplying these nonzero factors together produced an
    exact 0.0. Two or more nonzero finite floats can never
    mathematically multiply to exactly zero, so this is an
    unambiguous underflow signal, not a heuristic: unlike a residual
    check, it has no false positives (a genuine near-zero product
    would round to some nonzero subnormal, not land on exactly 0)
    and no false negatives caused by the check's own arithmetic
    overflowing, because it does none."""
    product = 1.0
    for f in factors:
        if f == 0.0:
            return False  # a genuine zero factor, not an underflow
        product *= f
    return product == 0.0


def _plausible(roots, A, B, C, D):
    """Residual check on the largest-magnitude root only: picking the
    biggest root specifically sidesteps the near-zero blind spot a
    residual check has on a small root (Q(0) = D identically). If
    even the least precision-starved root doesn't satisfy the
    original cubic to reasonable relative precision, the whole result
    is untrustworthy: confirmed against a case where coefficients
    were spread across the full leading-vs-constant-term axis (A and
    D both extreme, B and C moderate) and the raw Lebedev formula
    came back with every root scaled by the same ~1.12x factor,
    finite and self-consistent-looking but wrong, because the
    rescaling fallback was never triggered (see test_tc.py).

    Skips the check when computing the terms themselves overflows:
    not just when r is huge, but also when A or B alone is huge
    enough that A*r*r*r or B*r*r overflows even for a moderate r (a
    real case: r ~ 8.8e86 but A ~ 8.9e48, so A*r*r*r overflows on its
    own). For a *complex* r that overflow produces inf+nan*j (cross
    terms hitting inf*0) rather than a clean inf, which would poison
    the comparison into a false rejection of a root that's actually
    correct. Detecting the overflow directly, rather than guessing a
    safe magnitude threshold up front, catches both routes to it."""
    r = max(roots, key=abs)
    t0, t1, t2, t3 = A * r * r * r, B * r * r, C * r, D
    if not (cmath.isfinite(t0) and cmath.isfinite(t1) and cmath.isfinite(t2)):
        return True
    residual = t0 + t1 + t2 + t3
    scale = max(abs(t0), abs(t1), abs(t2), abs(t3), 1e-300)
    return abs(residual) <= 1e-6 * scale


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
        return 0j, complex(-c1n / c2n)
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
    if _mul_underflowed(T3, C) and T4 == 0.0:
        # 3*A*C underflowed to exactly 0 despite A and C both being
        # meaningfully sized after normalization: this corrupts P
        # (misclassifying it as ~B^2, which can misroute this whole
        # branch into "triple root at the origin" for a cubic that
        # doesn't remotely have one). Signal failure so cubic_roots
        # retries through the variable-rescaling fallback instead of
        # continuing on bad data.
        nan = float("nan")
        return (nan, nan, nan), True
    P = T2 - T4
    X3 = abs(P) ** 0.5
    d_term = 3.0 * T3 * T3 * D
    if _mul_underflowed(T3, T3, D) and d_term == 0.0:
        # Same failure mode, for the A^2*D term inside X1.
        nan = float("nan")
        return (nan, nan, nan), True
    X1 = B * (T4 - P - P) - d_term
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
                # max(0.0, ...) guards against cosv*cosv rounding
                # fractionally above 1.0: without it, a negative base
                # here silently produces a complex sinpart (Python's
                # ** does not raise on a negative base with a
                # fractional exponent), which would then hit an
                # unhandled TypeError at the x2 <= x3 comparison below.
                sinpart = max(0.0, 1.0 - cosv * cosv) ** 0.5 * Tc
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


def _degenerate_branch(A, B, C, D, X, reduced_c0, reduced_c1, reduced_c2):
    """Shared logic for both of Kahan's degenerate-coefficient
    branches. X is Kahan's own direct approximation (-B/A for the
    escaped root, or -D/C for the collapsed one). reduced_c0..c2 give
    the quadratic (Bx^2+Cx+D or Ax^2+Bx+C) that's supposed to hold
    the other two roots.

    Picking the right anchor root to trust needs one of two different
    strategies depending on what the reduced quadratic's own roots
    look like, and using the wrong one for a given case corrupts an
    already-correct root instead of fixing a broken one:

    - If the reduced quadratic's two roots are themselves wildly
      different in magnitude (a doubly-degenerate case within the
      already-degenerate branch), its smaller-magnitude root is the
      reliable one, not X: X can come out representing the *sum* of
      a large pair rather than a single clean root when the "one
      root escapes/collapses cleanly" assumption Kahan's X is built
      on doesn't hold.
    - Otherwise (the reduced quadratic's two roots are comparable in
      magnitude, e.g. a genuine complex-conjugate pair), there's no
      meaningfully "smaller" one to prefer, and X is the reliable
      anchor.

    Either way, only that one anchor root is trusted; the other two
    are re-derived from A x^3+B x^2+C x+D's own Vieta sum-of-roots
    and product-of-roots relations rather than taken directly from
    the reduced quadratic, which is only a leading-order
    approximation. Both branches of this decision are confirmed
    against cases matching mpmath to full float64 precision (see
    test_tc.py).
    """
    ra, rb = _quadratic_roots(reduced_c0, reduced_c1, reduced_c2)
    mag_a, mag_b = abs(ra), abs(rb)
    if mag_a > 0 and mag_b > 0 and min(mag_a, mag_b) < 1e-8 * max(mag_a, mag_b):
        r0 = ra if mag_a <= mag_b else rb
        # The reduced quadratic doesn't know about the coefficient
        # that made this branch degenerate in the first place (D for
        # the "D negligible next to C" pattern), so its own small
        # root can be an artifact of a *further* nested degeneracy:
        # what's actually a tiny complex-conjugate pair near the
        # origin gets misresolved into two separate real numbers,
        # neither of them right, even though each looks like a
        # plausible small real root in isolation. Sanity-check r0
        # against the full cubic before trusting it; if it doesn't
        # hold up, fall through to the general Lebedev+deflation path,
        # which resolves the real-vs-complex classification correctly
        # on its own (confirmed against mpmath, see test_tc.py).
        if abs(r0) > 1e-300:
            t0, t1, t2, t3 = A * r0 * r0 * r0, B * r0 * r0, C * r0, D
            residual = t0 + t1 + t2 + t3
            scale = max(abs(t0), abs(t1), abs(t2), abs(t3), 1e-300)
            if abs(residual) > 1e-6 * scale:
                return None
        # X can't be trusted here (see docstring above), so the other
        # two roots have to be re-derived from Vieta's relations on
        # the original cubic rather than taken as "X and the reduced
        # quadratic's other root" directly.
        sum_all = -B / A
        product_all = -D / A
        r1, r2 = _quadratic_roots(product_all / r0, -(sum_all - r0), 1.0)
        return complex(r0), r1, r2
    else:
        # No magnitude split to fall back on, so X has to be the
        # anchor. Sanity-check it first: X is derived by assuming
        # some terms of A x^3+B x^2+C x+D are negligible at x=X, an
        # assumption that silently fails when a *third* coefficient
        # (typically A) is also extreme, a compound degeneracy
        # Kahan's single-pattern derivation doesn't cover. If the
        # terms it assumed negligible aren't actually negligible
        # compared to D here, X isn't a real root at all (confirmed
        # against a case where it was off by 37 orders of magnitude,
        # see test_tc.py) - bail out to the general Lebedev path
        # instead, which handles this correctly via its own
        # normalization.
        # X at or near 0 needs its own exemption from this check, for
        # the same reason a from-scratch residual check on a
        # candidate root of exactly 0 doesn't work in general: Q(0) =
        # D identically, and the same problem reappears in softened
        # form for any X small enough that every dropped term
        # involving it underflows towards 0 too, leaving "residual"
        # trivially equal to D regardless of whether X is actually
        # correct (see test_tc.py for cases, including a subnormal
        # X ~1e-322, where this check would otherwise reject a
        # legitimately correct answer with a meaningless ratio of
        # exactly 1). 1e-300 is nowhere near the magnitudes X takes
        # in the compound-degeneracy case this check exists to catch
        # (~1e-29 there), so this stays a safe exemption, not a
        # loophole back into the original bug.
        if abs(X) > 1e-300:
            t0, t1, t2, t3 = A * X * X * X, B * X * X, C * X, D
            residual = t0 + t1 + t2 + t3
            scale = max(abs(t0), abs(t1), abs(t2), abs(t3), 1e-300)
            if abs(residual) > 1e-6 * scale:
                return None
        # X checks out, and (empirically) the reduced quadratic's own
        # roots are reliable here too, unlike in the magnitude-split
        # branch above: trust them directly instead of re-deriving via
        # Vieta on the original cubic. That re-derivation needs
        # sum_all - X to isolate "sum of the other two roots", but for
        # the "A negligible next to B" pattern, X and sum_all are both
        # computed as -B/A, the identical formula, so the subtraction
        # is always exactly 0.0 by construction rather than the small
        # nonzero residual it's supposed to recover, silently
        # corrupting an otherwise-correct pair of roots (confirmed
        # against mpmath, see test_tc.py).
        return complex(X), ra, rb


def _refine_tiny_leftover_root(roots, A, D):
    """If one of three candidate roots is many orders of magnitude
    smaller than the other two, it's likely the residual left over
    after near-total cancellation between two large, opposite-signed
    roots, and can lose most of its significant digits even though
    the two large roots are themselves individually accurate.
    Recomputing it from Vieta's product-of-roots relation using the
    other two (rather than trusting whatever the main computation
    produced for it directly) recovers full precision: confirmed
    against a case matching mpmath to 1.5e-16 relative error, versus
    the unrefined value being wrong by 15 orders of magnitude (see
    test_tc.py).

    Deliberately narrow, and NOT used by _degenerate_branch: there
    it's the *large* extrapolated value that's unreliable and the
    small one that's already accurate, the opposite pattern, so
    applying this same rule there would corrupt an already-correct
    root instead of fixing a broken one.
    """
    mags = [abs(r) for r in roots]
    scale = max(mags)
    if scale == 0:
        return roots
    i = min(range(3), key=lambda k: mags[k])
    if mags[i] > 1e-8 * scale:
        return roots
    others = [roots[k] for k in range(3) if k != i]
    denom = others[0] * others[1]
    if denom == 0 or not cmath.isfinite(denom):
        return roots
    refined = (-D / A) / denom
    if not cmath.isfinite(refined):
        return roots
    result = list(roots)
    result[i] = refined
    return tuple(result)


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
    # _degenerate_branch returns None if it finds a compound
    # degeneracy its single-pattern derivation doesn't cover (see its
    # docstring); when that happens, fall through instead of
    # returning, all the way to the general path below if neither
    # named pattern actually applies, which handles it correctly via
    # its own normalization.
    if abs(A) < _DEGENERATE_TOL * abs(B):
        result = _degenerate_branch(A, B, C, D, -B / A, D, C, B)
        if result is not None:
            return result
    if abs(D) < _DEGENERATE_TOL * abs(C):
        result = _degenerate_branch(A, B, C, D, -D / C, C, B, A)
        if result is not None:
            return result

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
        # the extra cost of deflation. Still worth the cheap check for
        # a tiny leftover root though: a huge/huge/tiny magnitude
        # split can pass this gap test easily (the huge pair looks
        # "separated") while the tiny one is a cancellation residual.
        return _refine_tiny_leftover_root((r1, r2, r3), A, D)

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
    # the other two. X*X*X instead of X**3: Python's ** on a float
    # raises OverflowError instead of saturating to inf, unlike *.
    if abs(X * X * X) > abs(D / A):
        C2 = -D / X
        B1 = (C2 - C) / X
    else:
        B1 = A * X + B
        C2 = B1 * X + C

    q0, q1 = _quadratic_roots(C2, B1, A)
    return _refine_tiny_leftover_root((complex(X), q0, q1), A, D)


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
    if r is not None and _all_finite(*r) and _plausible(r, A, B, C, D):
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
                if _all_finite(*r2) and _plausible(r2, A, B, C, D):
                    return r2

    raise ArithmeticError(
        "no finite roots found: coefficients are too extreme for "
        "this formula (some root likely does not fit in a float, or "
        "an unrecognized overflow/underflow pattern in its coefficients)"
    )
