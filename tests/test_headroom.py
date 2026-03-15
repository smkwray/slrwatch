from slr_watch.headroom import compute_actual_slr, compute_headroom, headroom_dollars


def test_compute_actual_slr():
    assert compute_actual_slr(55, 1000) == 0.055


def test_headroom_dollars():
    # Tier1 / req - exposure = 55 / 0.05 - 1000 = 100
    assert headroom_dollars(55, 0.05, 1000) == 100.0


def test_compute_headroom():
    result = compute_headroom(
        tier1_capital=55,
        total_leverage_exposure=1000,
        required_slr=0.05,
    )
    assert result.actual_slr == 0.055
    assert round(result.headroom_pp, 6) == 0.005
    assert result.headroom_dollars == 100.0
