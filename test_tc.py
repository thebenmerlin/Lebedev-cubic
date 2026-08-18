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
