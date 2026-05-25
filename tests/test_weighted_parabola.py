from swishsync_cv.config import ShotCandidateConfig
from swishsync_cv.data import SparseBallDetection
from swishsync_cv.tracking.parabola import fit_weighted_parabola, fitting_weight


def test_fitting_weight_tiers():
    assert fitting_weight(0.10) == 0.15
    assert fitting_weight(0.45) == 0.50
    assert fitting_weight(0.85) == 1.00


def test_weighted_parabola_prefers_high_confidence_points():
    noisy_low = [
        SparseBallDetection(0, 0.0, 10.0, 50.0, 0.15),
        SparseBallDetection(1, 33.3, 20.0, 40.0, 0.90),
        SparseBallDetection(2, 66.6, 30.0, 35.0, 0.92),
        SparseBallDetection(3, 99.9, 40.0, 40.0, 0.91),
    ]
    clean_high = [
        SparseBallDetection(0, 0.0, 10.0, 50.0, 0.90),
        SparseBallDetection(1, 33.3, 20.0, 40.0, 0.92),
        SparseBallDetection(2, 66.6, 30.0, 35.0, 0.91),
        SparseBallDetection(3, 99.9, 40.0, 40.0, 0.93),
    ]

    noisy_result = fit_weighted_parabola(noisy_low)
    clean_result = fit_weighted_parabola(clean_high)

    assert noisy_result is not None
    assert clean_result is not None
    noisy_fit, noisy_diag = noisy_result
    clean_fit, clean_diag = clean_result

    assert noisy_diag.outlier_count >= 0
    assert clean_diag.weighted_residual_rmse <= noisy_diag.weighted_residual_rmse
    assert clean_fit.weighted_r_squared >= noisy_fit.weighted_r_squared
