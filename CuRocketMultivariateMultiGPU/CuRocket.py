"""Rocket transformer using CUDA."""

# __author__ = ["angus924"]
# __all__ = ["Rocket"]


import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer


class CuRocketMultivariateMultiGPU(BaseTransformer):
    """RandOm Convolutional KErnel Transform (ROCKET).
       Is executed on GPU using CUDA.
    """

    _tags = {
        # packaging info
        # --------------
        # "authors": ["angus924"],
        # "maintainers": ["angus924"],
        # "python_dependencies": ["numba", "cupy"],
        # estimator type
        # --------------
        "univariate-only": False,
        "fit_is_empty": False,
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Primitives",
        # what is the scitype of y: None (not needed), Primitives, Series, Panel
        "scitype:instancewise": False,  # is this an instance-wise transform?
        "X_inner_mtype": "numpy3D",  # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
    }

    def __init__(self, num_kernels=10_000, normalise=True, random_state=None):# kernel_lengths=(7, 13, 2)):
        self.num_kernels = num_kernels
        self.normalise = normalise
        self.random_state = random_state if isinstance(random_state, int) else None
        # self.n_timepoints_min = n_timepoints_min
        # self.kernel_lengths = kernel_lengths
        # self.sums = None #O
        super().__init__()

    def _fit(self, X, y=None):
        from sktime.transformations.panel.rocket._rocket_numba import _generate_kernels

        if X.ndim == 3:
            n_instances, n_columns, n_timepoints = X.shape
        else:
            raise Exception("Input array has unsupported amount of dimensions")
        
        self.kernels = _generate_kernels(
            n_timepoints, self.num_kernels, n_columns, self.random_state
        )
        return self

    def _transform(self, X, y=None):

        from CuRocketMultivariateMultiGPU.curocket_multivariate_multigpu import _apply_kernels

        if self.normalise:
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, keepdims=True) + 1e-8
            )
        temp = _apply_kernels(X.astype(np.float32), self.kernels)
        t = pd.DataFrame(temp) # original

        return t
