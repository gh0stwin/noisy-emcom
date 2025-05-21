from noisy_emcom.utils import types


class Constant:
    def __call__(self, x):
        return x


class Linear:
    def __init__(self, m, b) -> None:
        self._m = m
        self._b = b

    def __call__(self, x):
        return self._m * x + self._b


def scaler_factory(scaler_type: types.ScalerType, kwargs: types.Config):
    if scaler_type == types.ScalerType.LINEAR:
        scaler = Linear(**kwargs)
    else:
        raise ValueError(f"Incorrect scaler type {scaler_type}")

    return scaler
