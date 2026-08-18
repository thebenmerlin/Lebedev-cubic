"""
Tests for tc.cubic_roots.

Ground truth for the randomized tests comes from mpmath (arbitrary
precision), used only here for validation. cubic_roots() itself has
no dependency on mpmath or numpy.

Run with: pytest test_tc.py
"""
from itertools import permutations

import mpmath as mp
import numpy as np
import pytest

from tc import cubic_roots

mp.mp.dps = 50


def mp_roots(A, B, C, D):
    A, B, C, D = mp.mpf(A), mp.mpf(B), mp.mpf(C), mp.mpf(D)
    return [complex(r) for r in
            mp.polyroots([D, C, B, A], maxsteps=1000, extraprec=800,
                         asc=True)]


def match_error(got, true):
    """Best-permutation max relative error between got and true."""
    got = [complex(g) for g in got]
    best = float("inf")
    for perm in permutations(range(3)):
        err = max(abs(got[perm[i]] - true[i]) / max(abs(true[i]), 1e-300)
                   for i in range(3))
        best = min(best, err)
    return best


# ---------------------------------------------------------------------
# Known-root sanity checks
# ---------------------------------------------------------------------

def test_three_distinct_real_roots():
    # (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
    roots = sorted(r.real for r in cubic_roots(1, -6, 11, -6))
    assert roots == pytest.approx([1, 2, 3])


def test_one_real_and_a_complex_pair():
    # x^3 - 1 = (x-1)(x^2+x+1)
    roots = cubic_roots(1, 0, 0, -1)
    reals = [r for r in roots if abs(r.imag) < 1e-9]
    pairs = [r for r in roots if abs(r.imag) >= 1e-9]
    assert len(reals) == 1 and reals[0].real == pytest.approx(1)
    assert len(pairs) == 2
    assert pairs[0] == pytest.approx(pairs[1].conjugate())
    assert pairs[0].real == pytest.approx(-0.5)


def test_repeated_root():
    # (x-1)^2 (x-2) = x^3 - 4x^2 + 5x - 2
    roots = sorted(r.real for r in cubic_roots(1, -4, 5, -2))
    assert roots == pytest.approx([1, 1, 2])


def test_triple_root_at_origin():
    roots = cubic_roots(1, 0, 0, 0)
    for r in roots:
        assert abs(r) < 1e-9


def test_a_must_be_nonzero():
    with pytest.raises(ValueError):
        cubic_roots(0, 1, 2, 3)


# ---------------------------------------------------------------------
# Regression tests for specific bugs found during development
# ---------------------------------------------------------------------

def test_overflow_from_squaring_a_large_middle_coefficient():
    # A near-double root where an early implementation's intermediate
    # computation overflowed even though every true root fits easily
    # in a float.
    A, B, C, D = -1.9759437903573098, 20.676028844161326, \
        -0.07239191557469431, 6.337612713131438e-05
    got = cubic_roots(A, B, C, D)
    true = mp_roots(A, B, C, D)
    assert match_error(got, true) < 1e-6


def test_near_double_root_boundary_needs_deflation():
    # This is the case that originally motivated deflation: the raw
    # Lebedev formula alone lands on the wrong side of the real-vs-
    # complex-pair classification right at this boundary and loses
    # several digits (~2.4e-5 relative error, one root near 10.46,
    # the other two clustered near 0.00175) even though the formula
    # visits the mathematically correct branch structure. Deflation
    # brings that down to ~1e-8. See README.md for the full story.
    A, B, C, D = -1.9759437903573098, 20.676028844161326, \
        -0.07239191557469431, 6.337612713131438e-05
    got = cubic_roots(A, B, C, D)
    true = mp_roots(A, B, C, D)
    assert match_error(got, true) < 1e-7


def test_extreme_magnitude_still_finds_correct_roots():
    # A, B, C, D span ~350 orders of magnitude combined. Needs the
    # Kahan degenerate-coefficient handling to avoid spurious
    # over/underflow.
    A, B, C, D = -4.3373685951460186e-17, -2.9007433597042315e-153, \
        -9.271693385654752e+136, 4.151519691857965e+213
    got = cubic_roots(A, B, C, D)
    true = mp_roots(A, B, C, D)
    assert match_error(got, true) < 1e-10


def test_genuinely_unrepresentable_roots_raise_not_silently_wrong():
    # When coefficients are spread so widely that a true root simply
    # doesn't fit in a float, cubic_roots must raise rather than
    # return a silently wrong (or non-finite) answer.
    with pytest.raises(ArithmeticError):
        cubic_roots(5.054715868950633e-66, 6.19390697947615e+285,
                    4.033899228025724e+75, -1.4759492198394687e+117)


def test_ac_underflow_does_not_silently_misroute_to_a_fake_triple_root():
    # Found by an independent security review before this shipped:
    # A and C individually looked fine after normalizing by the
    # overall max coefficient, but 3*A*C itself underflowed to
    # exactly 0.0, corrupting an intermediate term (P) and silently
    # routing this into the "triple root at the origin" branch for a
    # cubic that has nothing of the sort. It returned (0, huge,
    # -huge) with a residual on the order of D itself - not
    # approximately right, not a root at all. D here is close enough
    # to float64's limit (~8.6e307) that the rescaling fallback can't
    # fully recover a correct answer either, so the fixed, correct
    # behavior is to fail safe (raise) rather than guess.
    A, B, C, D = 1.0, -0.0, -3.6750000000000003e+205, \
        -8.575000000000001e+307
    with pytest.raises(ArithmeticError):
        cubic_roots(A, B, C, D)


def test_genuine_near_zero_roots_are_not_rejected():
    # A companion case to the one above: here 0.0 genuinely *is* the
    # correct float64 answer for two of the three roots (confirmed
    # against mpmath at 1200 digits of precision), not a misroute.
    # An earlier, overly blunt fix for the bug above (a residual
    # check applied uniformly to every root) rejected this
    # legitimate answer, since Q(0) = D identically regardless of
    # whether 0 is the right root, making that check meaningless
    # right at 0. Confirms that regression stays fixed.
    A, B, C, D = -1.561751275291703e+147, -1.6927153959491392e+281, \
        9.514815710074996e-105, -1.3264030610177571e-77
    got = cubic_roots(A, B, C, D)
    true = mp_roots(A, B, C, D)
    assert match_error(got, true) < 1e-6


def test_tiny_leftover_root_from_huge_pair_cancellation():
    # Found by a systematic sweep over all 6 pairs of coefficients
    # pushed to extreme, uncorrelated magnitudes: two roots land at
    # ~6e53 and ~-6e53, and the third (true value ~-1.77e-4) is what's
    # left over after they nearly cancel. The unrefined computation
    # got that leftover root wrong by 15 orders of magnitude even
    # though the two large roots were individually accurate to full
    # precision; recomputing it from Vieta's product-of-roots
    # relation using the other two fixes it.
    A, B, C, D = -1.2046889379604413e-106, -1.202895591388265e-130, \
        44.26347838804574, 0.007826638382920238
    got = cubic_roots(A, B, C, D)
    true = mp_roots(A, B, C, D)
    assert match_error(got, true) < 1e-9


def test_compound_degeneracy_where_a_is_also_extreme():
    # D negligible next to C fires correctly here, but A is *also*
    # absurdly large relative to C (a compound degeneracy Kahan's
    # single-pattern derivation doesn't cover), which makes X = -D/C
    # not an actual root at all: Q(X) came out ~1e38 away from zero,
    # not approximately right. Falling through to the general Lebedev
    # path instead (which handles it correctly via its own
    # normalization) fixes it.
    A, B, C, D = 3.9569641531183267e+124, -0.05581608706392034, \
        5.483624929179232e+29, 10.047002261075267
    got = cubic_roots(A, B, C, D)
    true = mp_roots(A, B, C, D)
    assert match_error(got, true) < 1e-9


def test_subnormal_x_is_not_rejected_by_the_compound_degeneracy_check():
    # A companion case to the one above: X here is a subnormal
    # number (~1e-322), and 0 genuinely is (to within float64's
    # resolution) the correct answer for that root. The compound-
    # degeneracy sanity check on X has the same Q(0)-is-uninformative
    # blind spot as the very first fix in this file when X is at or
    # near 0, and needs the same kind of exemption.
    A, B, C, D = -1.9192995433020904e+44, 1.1331553724392622e-212, \
        9.619131160888853e+267, 3.5631818321321395e-119
    got = cubic_roots(A, B, C, D)
    true = mp_roots(A, B, C, D)
    assert match_error(got, true) < 1e-6


# ---------------------------------------------------------------------
# Randomized correctness, against mpmath ground truth
# ---------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_random_cubics_match_mpmath(seed):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        scale = 10.0 ** rng.uniform(-4, 4)
        if rng.random() < 0.5:
            roots = list(rng.standard_normal(3) * scale)
        else:
            re = rng.standard_normal() * scale
            im = rng.standard_normal() * scale
            r3 = rng.standard_normal() * scale
            roots = [complex(re, im), complex(re, -im), r3]
        A = rng.choice([-1, 1]) * 10.0 ** rng.uniform(-4, 4)
        ra, rb, rc = roots
        B = -A * (ra + rb + rc)
        C = A * (ra * rb + ra * rc + rb * rc)
        D = -A * ra * rb * rc
        A, B, C, D = (float(np.real(x)) for x in (A, B, C, D))

        got = cubic_roots(A, B, C, D)
        true = mp_roots(A, B, C, D)
        assert match_error(got, true) < 1e-8, (A, B, C, D)


@pytest.mark.parametrize("seed", range(10))
def test_random_clustered_roots_match_mpmath(seed):
    """Roots deliberately placed close together, the regime deflation
    exists for."""
    rng = np.random.default_rng(1000 + seed)
    for _ in range(20):
        scale = 10.0 ** rng.uniform(-4, 4)
        r1 = rng.standard_normal() * scale
        closeness = 10.0 ** rng.uniform(-12, -2)
        r2 = r1 + scale * closeness * rng.choice([-1, 1])
        r3 = rng.standard_normal() * scale * 10.0 ** rng.uniform(-2, 2)
        A = rng.choice([-1, 1]) * 10.0 ** rng.uniform(-4, 4)
        B = -A * (r1 + r2 + r3)
        C = A * (r1 * r2 + r1 * r3 + r2 * r3)
        D = -A * r1 * r2 * r3

        got = cubic_roots(A, B, C, D)
        true = mp_roots(A, B, C, D)
        # looser tolerance here: near-coincident roots are inherently
        # harder to resolve to full precision for any method (see
        # README.md, this matches Kahan's own accuracy analysis).
        assert match_error(got, true) < 1e-4, (A, B, C, D)


def test_always_returns_complex_not_bare_float():
    # Found by an independent security review: _degenerate_branch's
    # anchor-root fallback returned X (a plain float) unwrapped in
    # two of its return statements, so cubic_roots could hand back a
    # tuple mixing float and complex entries - a silent violation of
    # the documented "tuple of 3 complex numbers" contract (harmless
    # numerically since float and complex interoperate, but would
    # surprise any caller doing isinstance()/type() dispatch).
    cases = [
        (-1.561751275291703e+147, -1.6927153959491392e+281,
         9.514815710074996e-105, -1.3264030610177571e-77),
        (-1.9192995433020904e+44, 1.1331553724392622e-212,
         9.619131160888853e+267, 3.5631818321321395e-119),
    ]
    for A, B, C, D in cases:
        for r in cubic_roots(A, B, C, D):
            assert isinstance(r, complex), (A, B, C, D, type(r))


def test_never_returns_non_finite_without_raising():
    """Across a wide range of coefficient magnitudes, cubic_roots
    must either return finite roots or raise ArithmeticError/
    ValueError. It must never return inf/nan silently."""
    rng = np.random.default_rng(7)
    for _ in range(2000):
        exps = rng.uniform(-250, 250, size=4)
        c = rng.choice([-1, 1], size=4) * rng.uniform(1, 10, size=4) \
            * 10.0 ** exps
        A, B, C, D = c.astype(float)
        if A == 0:
            continue
        try:
            got = cubic_roots(A, B, C, D)
        except (ArithmeticError, ValueError):
            continue
        for r in got:
            assert np.isfinite(r.real) and np.isfinite(r.imag), \
                (A, B, C, D, got)
