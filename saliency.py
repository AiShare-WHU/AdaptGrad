import copy
import numpy as np
import torch as tc
import torchvision as tv
import torch.nn as nn
from core import Gradient, Mask, Sigma, Bound, Baseline


class VanillaGradient(Gradient):
    def __init__(self):
        super(VanillaGradient, self).__init__()
        self.name = "Vanilla" + self.name

    def gradient(self, model_call, x, label):
        if isinstance(x, np.ndarray):
            x = tc.tensor(x, dtype=tc.float32, device=self.device)
        elif isinstance(x, tc.Tensor) and x.is_cpu:
            x = x.to(self.device)
        x.requires_grad = True
        prediction = model_call(x)
        loss = prediction[:, label]
        loss.backward()
        gradient = x.grad
        return gradient


class GaussianMask(Mask):
    def __init__(self, sigma):
        super(GaussianMask, self).__init__(sigma)
        self.name = "Gaussian" + self.name

    def get_mask(self, x) -> tc.Tensor:
        shape = x.shape
        sigma = self.sigma.get_sigma(x)
        mask = tc.randn(shape, device=self.device) * sigma
        return mask


class FixedSigma(Sigma):
    def __init__(self, alpha, bound):
        """
        @param alpha: Proportion of noise, generally 0.2-0.3.
        @param bound: The range of input data, bound [0] is the lower bound, and bound [1] is the upper bound.
        """
        super(FixedSigma, self).__init__(alpha, bound)
        self.name = "Fixed" + self.name

    def get_sigma(self, x):
        """
        @param x: Input data.
        @return: sigma
        @note: Only valid for Gaussian kernel
        """
        lower, upper = self.bound.get_bound(x)
        return (lower - upper) * self.alpha


class AdaptedGaussianSigma(Sigma):
    def __init__(self, alpha, bound):
        """
        @param alpha: Confidence level, generally 0.95, 0.99, 0.995, etc.
        @param bound: Confidence interval, bound [0] is the lower bound, and bound [1] is the upper bound.
        """
        super(AdaptedGaussianSigma, self).__init__(alpha, bound)
        self.name = "AdaptedGaussian" + self.name

    def get_sigma(self, x):
        """
        @param x: Input data.
        @return: sigma
        """
        lower, upper = self.bound.get_bound(x)
        t = (upper - lower) / 2
        if isinstance(t, np.ndarray) or isinstance(t, float):
            t = tc.tensor(t, device=self.device)
        alpha = tc.tensor(self.alpha, device=self.device)
        sigma = t / (np.sqrt(2) * tc.erfinv(alpha))
        return sigma


class FixedBound(Bound):
    def __init__(self, lower, upper):
        super(FixedBound, self).__init__(lower, upper)
        self.name = "Fixed" + self.name

    def get_bound(self, x):
        return self.lower, self.upper


class AdaptedBound(Bound):
    def __init__(self, lower, upper):
        super(AdaptedBound, self).__init__(lower, upper)
        self.name = "Adapted" + self.name

    def get_bound(self, x):
        if isinstance(x, np.ndarray):
            x = tc.tensor(x, dtype=tc.float32, device=self.device)
        else:
            x = x.to(self.device)
        min_abs_value = tc.min(tc.abs(x - self.lower), tc.abs(x - self.upper))
        return -min_abs_value, min_abs_value


class SmoothGradient(Gradient):
    def __init__(self, mask: Mask, n_samples: int):
        super(SmoothGradient, self).__init__()
        self.mask = mask
        self.n_samples = n_samples
        self.name = "Smooth" + self.name + mask.name

    def gradient(self, model_call, x, label):
        self.mask.to(self.device)
        if isinstance(x, np.ndarray):
            x = tc.tensor(x, dtype=tc.float32, device=self.device)
        else:
            x = x.to(self.device)
        x = x.detach()
        x.requires_grad = True
        prediction = model_call(x)
        loss = prediction[:, label]
        loss.backward()
        grads = tc.zeros(size=[self.n_samples + 1, *x.shape], device=self.device)
        grads[0] = x.grad  # type: ignore
        for i in range(self.n_samples):
            mask = self.mask.get_mask(x)
            x_noise = x + mask
            x_noise = x_noise.detach()
            x_noise.requires_grad = True
            prediction = model_call(x_noise)
            loss = prediction[:, label]
            loss.backward()
            grads[i + 1] = x_noise.grad  # type: ignore
        grads = grads[~tc.isnan(grads.view(grads.shape[0], -1)).any(dim=1)]
        return grads.mean(dim=0)


class WhiteBaseline(Baseline):
    def __init__(self):
        super(WhiteBaseline, self).__init__()
        self.name = "White" + self.name

    def get_baseline(self, x):
        return tc.zeros_like(x, device=self.device)


class BlackBaseline(Baseline):
    def __init__(self):
        super(BlackBaseline, self).__init__()
        self.name = "Black" + self.name

    def get_baseline(self, x):
        return tc.ones_like(x, device=self.device)


class IntegratedGradient(Gradient):
    def __init__(self, baseline: Baseline, backgrad: Gradient, n_steps: int):
        super(IntegratedGradient, self).__init__()
        self.name = "Integrated" + self.name + baseline.name + backgrad.name
        self.baseline = baseline
        self.backgrad = backgrad
        self.n_steps = n_steps

    def gradient(self, model_call, x, label):
        self.baseline.to(self.device)
        self.backgrad.to(self.device)
        if isinstance(x, np.ndarray):
            x = tc.tensor(x, dtype=tc.float32, device=self.device)
        else:
            x = x.to(self.device)
        x = x.detach()
        x.requires_grad = True
        prediction = model_call(x)
        loss = prediction[:, label]
        loss.backward()
        grads = tc.zeros(size=[self.n_steps + 1, *x.shape], device=self.device)
        grads[0] = x.grad  # type: ignore
        x_baseline = self.baseline.get_baseline(x)
        for i in range(self.n_steps):
            alpha = i / self.n_steps
            x_interpolated = x_baseline + alpha * (x - x_baseline)
            x_interpolated = x_interpolated.detach()
            grads[i + 1] = self.backgrad.gradient(model_call, x_interpolated, label)
        grads = grads[~tc.isnan(grads.view(grads.shape[0], -1)).any(dim=1)]
        ig = grads.mean(dim=0) * (x - x_baseline)
        return ig.detach()


class GradientxInput(Gradient):
    def __init__(self, backgrad: Gradient):
        super(GradientxInput, self).__init__()
        self.name = "GradientxInput" + self.name + backgrad.name
        self.backgrad = backgrad

    def gradient(self, model_call, x, label):
        self.backgrad.to(self.device)
        if isinstance(x, np.ndarray):
            x = tc.tensor(x, dtype=tc.float32, device=self.device)
        else:
            x = x.to(self.device)
        grad = self.backgrad.gradient(model_call, x, label)
        gi = grad * x
        return gi.detach()


class NoiseGradient(Gradient):
    def __init__(self, noise: Mask, n_samples: int, backgrad: Gradient):
        super(NoiseGradient, self).__init__()
        self.noise = noise
        self.name = "NoiseGrad" + self.name + noise.name + backgrad.name
        self.n_samples = n_samples
        self.backgrad = backgrad

    def gradient(self, model_call, x, label):
        if isinstance(x, np.ndarray):
            x = tc.tensor(x, dtype=tc.float32, device=self.device)
        else:
            x = x.to(self.device)
        # Check model_call(x) is a nn.Module
        if not isinstance(model_call, nn.Module):
            raise ValueError("model_call must be a nn.Module")
        # Copy the model
        model_copy = copy.deepcopy(model_call)
        model_copy.to(self.device)
        grads = tc.zeros(size=[self.n_samples + 1, *x.shape], device=self.device)
        x = x.detach()
        x.requires_grad = True
        prediction = model_call(x)
        loss = prediction[:, label]
        loss.backward()
        grads[0] = x.grad  # type: ignore
        for i in range(self.n_samples):
            model_copy.load_state_dict(model_call.state_dict())
            with tc.no_grad():
                for name, param in model_copy.named_parameters():
                    if "classifier" in name:
                        continue
                    param *= self.noise.get_mask(param) + 1
            grads[i + 1] = self.backgrad.gradient(model_call, x, label)
        grads = grads[~tc.isnan(grads.view(grads.shape[0], -1)).any(dim=1)]
        return grads.mean(dim=0)
