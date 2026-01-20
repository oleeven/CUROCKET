import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class CuRocketUnivariate(BaseTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET).
       Is executed on GPU using CUDA.
    """

    _tags = {
        "univariate-only": True,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
    }

    def __init__(self, num_kernels=10_000, normalise=True, random_state=None):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.random_state = random_state if isinstance(random_state, int) else None
        super().__init__()

    def _fit(self, X, y=None):
        from sktime.transformations.panel.rocket._rocket_numba import _generate_kernels
        
        if X.ndim == 2:
            n_instances, n_timepoints = X.shape
            n_columns = 1
        elif X.ndim == 3:
            n_instances, n_columns, n_timepoints = X.shape
            if n_columns != 1:
                raise Exception("Input array has unsupported amount of channels")
        else:
            raise Exception("Input array has unsupported amount of dimensions")

        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, n_columns, self.random_state
        )
        return self

    def _transform(self, X, y=None):
        from CuRocketUnivariate.curocket_univariate import _apply_kernels_univariate

        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        t = pd.DataFrame(_apply_kernels_univariate(X.astype(np.float32), self.kernels))

        return t
