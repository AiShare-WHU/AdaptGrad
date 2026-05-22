import os
from typing import List, Union
from matplotlib import pyplot as plt
import torch as tc
import seaborn as sns
import torch.nn as nn
import numpy as np
import scipy.stats as stats
from core import Bound, Sigma, Mask, Gradient, Baseline


def extract_layers(model: nn.Module) -> List[nn.Module]:
    layers = []
    children = list(model.children())
    if not children:
        return [model]
    else:
        for child in children:
            try:
                layers.extend(extract_layers(child))
            except TypeError:
                layers.append(child)
        return layers


def get_salience(
    model: nn.Module,
    explainer: Gradient,
    img: tc.Tensor,
    class_id: int,
    index: Union[int, str] = 0,
    path: str = "./saved_data/imagenet",
    pin: bool = True,
) -> np.ndarray:
    path = path + "/" + explainer.name
    if not pin:
        salience = explainer.gradient(model, img, class_id)
        if salience.is_cuda:
            salience = salience.cpu().detach().numpy()
        else:
            salience = salience.detach().numpy()
        salience = salience[0]
        salience = np.sum(salience, axis=0)
        return salience
    else:
        if os.path.exists(path + "/" + str(index) + ".npy"):
            salience = np.load(path + "/" + str(index) + ".npy")
        else:
            if not os.path.exists(path + "/"):
                os.makedirs(path + "/")
            salience = explainer.gradient(model, img, class_id)
            if salience.is_cuda:
                salience = salience.cpu().detach().numpy()
            else:
                salience = salience.detach().numpy()
            salience = salience[0]
            if len(salience.shape) == 3:
                salience = np.sum(salience, axis=0)
            np.save(path + "/" + str(index) + ".npy", salience)
        return salience


def visualize_absscale(salience):
    vmax = abs(np.percentile(salience, 99))
    vmin = np.min(salience)
    return np.clip((salience - vmin) / (vmax - vmin), 0, 1)


def visualize_noabsscale(salience):
    span = abs(np.percentile(salience, 99))
    vmin = -span
    vmax = span
    return np.clip(salience / (vmax - vmin), -1, 1)


def rank_correlation(saliency, target):
    saliency = saliency.flatten()
    target = target.flatten()
    return stats.spearmanr(saliency, target)


def visualize_ixs(salience, input_img):
    # Multiplying maps with the input images
    salience = salience * input_img
    # Normalizing the salience maps
    salience = visualize_absscale(salience)
    return salience


def visualize_positive(salience):
    salience = np.maximum(0, salience)
    salience = visualize_absscale(salience)
    return salience


def visualize_negative(salience):
    salience = np.minimum(0, salience)
    salience = visualize_absscale(salience)
    return salience


def visualize_square(salience):
    salience = salience**2
    salience = visualize_absscale(salience)
    return salience


def set_seed(seed: int):
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)
    tc.backends.cudnn.deterministic = True
    tc.backends.cudnn.benchmark = False


def visual_imagenet(salience: tc.Tensor):
    if salience.is_cuda:
        salience = salience.cpu()
    salience = salience.detach().numpy()[0]
    vis_salience = np.sum(salience, axis=0)
    vis_salience = visualize_noabsscale(vis_salience)
    ax, fig = plt.subplots(1, 1, figsize=(10, 10))
    fig.imshow(vis_salience, alpha=1, cmap="coolwarm")
    fig.axis("off")
    plt.show()


def visual_mnist(salience: tc.Tensor):
    if salience.is_cuda:
        salience = salience.cpu()
    vis_salience = salience.view(28, 28).detach().numpy()
    vis_salience = visualize_absscale(vis_salience)
    ax, fig = plt.subplots(1, 1, figsize=(10, 10))
    sns.heatmap(vis_salience, cmap="gray", cbar=False, ax=fig)
    fig.axis("off")
    plt.show()


def array_to_tensor_imagenet(array: np.ndarray):
    tensor = tc.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.float()
    tensor = tensor / 255
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    std = tc.tensor(std).reshape(3, 1, 1)
    mean = tc.tensor(mean).reshape(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def tensor_to_array_imagenet(tensor: tc.Tensor):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    array = tensor.detach().numpy()
    mean = [0.485, 0.456, 0.406]
    mean = np.array(mean).reshape(3, 1, 1)
    std = [0.229, 0.224, 0.225]
    std = np.array(std).reshape(3, 1, 1)
    array = array * std + mean
    array = array - array.min()
    array = array / array.max()
    array = array * 255
    array = array.astype(np.uint8)
    array = array.transpose(1, 2, 0)
    return array


def model_call(model, device="cpu"):
    def call(img):
        tensor = array_to_tensor_imagenet(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(device)
        prediction = model(tensor).squeeze(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        result = score
        return result

    return call


def salience_to_array(salience: tc.Tensor):
    if salience.is_cuda:
        salience = salience.cpu()
    salience = salience.detach().numpy()[0]
    salience = np.sum(salience, axis=0)
    salience = visualize_absscale(salience)
    return salience
