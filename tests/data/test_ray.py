import pytest
import warnings
warnings.filterwarnings("ignore")

import ray

@pytest.mark.run(order=1)
def test_ray():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    @ray.remote
    def f(x):
        return x ** 2

    x_id = f.remote(2.)
    y_id = f.remote(3.)

    results = ray.get([x_id, y_id])

    ray.shutdown()

    assert results[0] == pytest.approx(4.)
    assert results[1] == pytest.approx(9.)