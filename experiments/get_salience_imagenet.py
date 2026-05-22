import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torchvision as tv
import torch as tc
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
from models import create_model
from utils import get_salience, set_seed

from sacred import Experiment

# from sacred.observers import SlackObserver

ex = Experiment("Salience")
# slack_obs = SlackObserver.from_config("./slack.json")
# ex.observers.append(slack_obs)


@ex.config
def config():
    dataset = "imagenet"  # only imagenet
    model_name = (
        "vgg16"  # vgg16, resnet50, densenet121, inception_v3, mobilenet_v3_small
    )
    n_samples = 50  # number of samples for sampling
    m_samples = 50  # number of samples for sampling (only for NoiseGradient)
    n_steps = 50  # number of steps for integrated gradient
    adapt_alpha = 0.95  # confidence level, 0.95, 0.99, 0.995, 0.999
    alpha = 0.2  # alpha for SmoothGradient
    samples = 100  # number of samples for salience map
    device_id = 0  # GPU device id
    seed = 0


@ex.automain
def main(
    dataset,
    model_name,
    n_samples,
    m_samples,
    n_steps,
    adapt_alpha,
    alpha,
    samples,
    device_id,
    seed,
):
    set_seed(seed)
    if tc.cuda.is_available():
        print("CUDA is available")
        device = tc.device("cuda:%d" % device_id)
    else:
        print("CUDA is not available")
        device = tc.device("cpu")
    model = create_model(model_name, device=device, eval_mode=True)
    saved_data_path = f"./saved_data/{model_name}/imagenet/"
    imgs_path = "./data/imagenet/images/"
    img_ids = [i.split(".")[0] for i in os.listdir(imgs_path)]
    img_ids.sort()
    bound_min = -2.117904
    bound_max = 2.640000
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

    for img_id in tqdm(img_ids):
        img_input = tv.io.read_image(imgs_path + img_id + ".JPEG")
        img_input = model.transform(img_input)
        img_input = img_input.unsqueeze(0)
        img_input = img_input.to(device)
        prediction = model(img_input).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        for saliency_method in saliency_methods:
            salience_map = get_salience(
                model=model,
                explainer=saliency_method.to(device),
                img=img_input,
                class_id=class_id,
                index=img_id,
                path=saved_data_path,
            )
            # print(img_id, saliency_method.name, salience_map.shape)shape
