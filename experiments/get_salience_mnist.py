import os
import sys

import PIL.Image as Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torchvision as tv
import torch as tc
import numpy as np
from tqdm import tqdm
from saliency import (
    FixedBound,
    AdaptedBound,
    FixedSigma,
    AdaptedGaussianSigma,
    GaussianMask,
    VanillaGradient,
    SmoothGradient,
    NoiseGradient,
    IntegratedGradient,
    GradientxInput,
    WhiteBaseline,
    BlackBaseline,
)
from models import MLP
from utils import get_salience

from sacred import Experiment

# from sacred.observers import SlackObserver

ex = Experiment("Salience")
# slack_obs = SlackObserver.from_config("./slack.json")
# ex.observers.append(slack_obs)


@ex.config
def config():
    dataset = "mnist"  # only mnist
    model_name = "MLP"  # only MLP
    kind = "Normal"  # Normal, Bias, Random
    n_samples = 50  # number of samples for sampling
    m_samples = 50  # number of samples for sampling (only for NoiseGradient)
    n_steps = 50  # number of steps for integrated gradient
    adapt_alpha = 0.95  # confidence level, 0.95, 0.99, 0.995, 0.999
    alpha = 0.2  # alpha for SmoothGradient
    samples = 1000  # number of samples for salience map
    device_id = 0


@ex.automain
def main(
    dataset,
    model_name,
    kind,
    n_samples,
    m_samples,
    n_steps,
    adapt_alpha,
    alpha,
    samples,
    device_id,
):
    if tc.cuda.is_available():
        print("CUDA is available")
        device = tc.device("cuda:%d" % device_id)
    else:
        print("CUDA is not available")
        device = tc.device("cpu")
    if kind == "Normal" or kind == "Random":
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
            ]
        )
    elif kind == "Bias":
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[-1] * 1, std=[1] * 1),
            ]
        )
    else:
        raise NotImplementedError

    train_data = tv.datasets.MNIST(
        "./data",
        train=True,
        transform=transform,
        download=True,
    )
    test_data = tv.datasets.MNIST(
        "./data",
        train=False,
        transform=transform,
        download=True,
    )
    train_loader = tc.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = tc.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    model = MLP()

    if kind == "Normal":
        model.load("./saved_models/mlp.pth", device=device)
    elif kind == "Bias":
        model.load("./saved_models/bias_mlp.pth", device=device)
    elif kind == "Random":
        model.load("./saved_models/random_mlp.pth", device=device)
    else:
        raise NotImplementedError
    model = model.to(device)
    # model.eval()
    saved_data_path = f"./saved_data/{model_name}/{dataset}{kind}/"
    bound_min = 0
    bound_max = 1
    fixed_bound = FixedBound(bound_min, bound_max).to(device)
    adapted_bound = AdaptedBound(bound_min, bound_max).to(device)
    noise_bound = FixedBound(0, 1).to(device)
    fixed_sigma = FixedSigma(alpha, fixed_bound).to(device)
    adapted_sigma = AdaptedGaussianSigma(adapt_alpha, adapted_bound).to(device)
    noise_sigma = FixedSigma(0.2, noise_bound).to(device)
    fixed_mask = GaussianMask(fixed_sigma).to(device)
    adapted_mask = GaussianMask(adapted_sigma).to(device)
    noise_mask = GaussianMask(noise_sigma).to(device)
    saliency_methods = [
        VanillaGradient(),
        SmoothGradient(fixed_mask, n_samples),
        SmoothGradient(adapted_mask, n_samples),
        NoiseGradient(noise_mask, m_samples, VanillaGradient().to(device)),
        NoiseGradient(
            noise_mask, m_samples, SmoothGradient(fixed_mask, n_samples).to(device)
        ),
        NoiseGradient(
            noise_mask, m_samples, SmoothGradient(adapted_mask, n_samples).to(device)
        ),
        IntegratedGradient(
            WhiteBaseline().to(device), VanillaGradient().to(device), n_steps
        ),
        IntegratedGradient(
            BlackBaseline().to(device), VanillaGradient().to(device), n_steps
        ),
        IntegratedGradient(
            WhiteBaseline().to(device),
            SmoothGradient(fixed_mask, n_samples).to(device),
            n_steps,
        ),
        IntegratedGradient(
            BlackBaseline().to(device),
            SmoothGradient(fixed_mask, n_samples).to(device),
            n_steps,
        ),
        IntegratedGradient(
            WhiteBaseline().to(device),
            SmoothGradient(adapted_mask, n_samples).to(device),
            n_steps,
        ),
        IntegratedGradient(
            BlackBaseline().to(device),
            SmoothGradient(adapted_mask, n_samples).to(device),
            n_steps,
        ),
        GradientxInput(VanillaGradient().to(device)),
        GradientxInput(SmoothGradient(fixed_mask, n_samples).to(device)),
        GradientxInput(SmoothGradient(adapted_mask, n_samples).to(device)),
    ]

    np.random.seed(0)
    indexs = np.random.choice(list(range(len(test_data))), size=samples)
    for index in tqdm(indexs):
        img, target = test_data[index][0][0], int(test_data[index][1])
        img = Image.fromarray(img.numpy(), mode="L")
        img = transform(img)
        img = img.to(device)
        img = img.view(img.shape[0], -1)
        prediction = model(img).squeeze().argmax().item()

        for saliency_method in saliency_methods:
            salience_map = get_salience(
                model=model,
                explainer=saliency_method.to(device),
                img=img,
                class_id=prediction,
                index=index,
                path=saved_data_path,
            )
