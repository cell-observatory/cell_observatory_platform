import pytest
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from data.quality_metrics import statistical_metrics

@pytest.mark.run(order=1)
def test_statistical_metrics():
    a = np.arange(10)
    mn, mx, mean, variance, skewness, kurtosis, differential_entropy = statistical_metrics(a)
    assert mn == 0
    assert mx == 9
    assert mean == pytest.approx(4.5)
    assert variance == pytest.approx(9.166666666666666)
    assert skewness == pytest.approx(0.)
    assert kurtosis == pytest.approx(-1.2242424242424244)
    assert differential_entropy == pytest.approx(2.0463983239016326)