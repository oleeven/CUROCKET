# CUROCKET
This repo contains a CUDA-based implementation of the feature-extraction algorithm ROCKET (RandOm Convolutional KErnel Transform) [[1](https://arxiv.org/abs/1910.13051)] to run ROCKET on GPU. On the test setup, it has up to 11 times higher computational efficiency per watt than ROCKET on CPU. CUROCKET is explained in details in this paper [[2](https://arxiv.org/)].

## Content
- CuRocketUnivariate: univariate version of CUROCKET
- CuRocketMultivariate: multivariate version of CUROCKET
- CuRocketMultivariateMultiGPU: multivariate version of CUROCKET that works on multiple GPUs in parallel. It has more overhead than single GPU, therefore it is only more efficient at a certain dataset size. Currently only mutltiple GPUs of the same type are supported

## Setup
You need a system with a CUDA-capable GPU and CUDA installed. The python packages sktime, numba and cupy need to be installed. If you want to use the notebook, install jupyter. Refer to the packages' documentation for a detailed explanation on how to install them. They can be installed via pip. Depending on your CUDA version, you have to install a different version of cupy. In the following example its for CUDA 12.x.
```bash
pip install sktime
pip install numba
pip install cupy-cuda12x
pip install jupyter
```

## Usage
The modules are set up in an sklearn-compatible format, so you can use functions fit and transform on an instance of CUROCKET as demonstrated in the following example.

```python
import numpy as np
from CuRocketUnivariate.CuRocket import CuRocketUnivariate

X = np.random.random((100, 1, 1000))
cuda_rocket = CuRocketUnivariate()
cuda_rocket.fit(X)
X_t = cuda_rocket.transform(X)
```

You can also use [this](example_usage_and_comparison.ipynb) notebook to test out the different versions and compare the calculation time.

## Citation
If you use this code, please cite:

```bibtex
@article{stueven_etal_2026,
  author={Stüven, Ole and Moenck, Keno and Schüppstuhl, Thorsten},
  title={CUROCKET: Optimizing ROCKET for GPU},
  year={2026},
}
```