from typing import Callable, Union, Tuple
import numpy as np
import torch as tc


class GPU:
    def __init__(self):
        self.device = tc.device("cpu")

    def to(self, device: tc.device):
        self.device = device
        return self

    def cuda(self):
        self.device = tc.device("cuda")
        return self.to(self.device)


class Gradient(GPU):
    def __init__(self):
        super(Gradient, self).__init__()
        self.name = "Gradient"

    def gradient(
        self,
        model_call: Callable[[tc.Tensor], tc.Tensor],
        x: Union[tc.Tensor, np.ndarray],
        label: int,
    ) -> tc.Tensor:
        raise NotImplementedError


class Bound(GPU):
    def __init__(
        self,
        lower: Union[float, np.ndarray, tc.Tensor],
        upper: Union[float, np.ndarray, tc.Tensor],
    ):
        super(Bound, self).__init__()
        self.lower = lower
        self.upper = upper
        self.name = "Bound"

    def get_bound(
        self, x: Union[np.ndarray, tc.Tensor]
    ) -> Tuple[
        Union[float, np.ndarray, tc.Tensor], Union[float, np.ndarray, tc.Tensor]
    ]:
        raise NotImplementedError

    def to(self, device: tc.device):
        self.device = device
        if isinstance(self.lower, np.ndarray):
            self.lower = tc.tensor(self.lower, dtype=tc.float32, device=device)
            self.upper = tc.tensor(self.upper, dtype=tc.float32, device=device)
        elif isinstance(self.lower, float):
            self.lower = tc.tensor(self.lower, device=device)
            self.upper = tc.tensor(self.upper, device=device)
        elif isinstance(self.lower, tc.Tensor):
            self.lower = self.lower.to(device)
            self.upper = self.upper.to(device)
        return self


class Sigma(GPU):
    def __init__(self, alpha: Union[float, np.ndarray, tc.Tensor], bound: Bound):
        super(Sigma, self).__init__()
        self.name = "Sigma" + bound.name
        self.alpha = alpha
        self.bound = bound

    def get_sigma(
        self, x: Union[np.ndarray, tc.Tensor]
    ) -> Union[float, np.ndarray, tc.Tensor]:
        raise NotImplementedError


class Mask(GPU):
    def __init__(self, sigma: Sigma):
        super(Mask, self).__init__()
        self.name = "Mask" + sigma.name
        self.sigma = sigma

    def get_mask(
        self,
        x: tc.Tensor,
    ) -> tc.Tensor:
        raise NotImplementedError


class Baseline(GPU):
    def __init__(self):
        super(Baseline, self).__init__()
        self.name = "Baseline"

    def get_baseline(
        self,
        x: tc.Tensor,
    ) -> tc.Tensor:
        raise NotImplementedError
