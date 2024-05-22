import copy
import numpy as np
import scipy.stats as stats
import torch


def extract_layers(model):
    layers = []
    children = list(model.children())
    if not children:
        return model
    else:
        for child in children:
            try:
                layers.extend(extract_layers(child))
            except TypeError:
                layers.append(child)
        return layers


def layer_randomization(model, layer_index):
    layers = extract_layers(model)
    if (
        "weight" in layers[layer_index].state_dict().keys()
        and layers[layer_index].weight.requires_grad
    ):
        torch.nn.init.normal_(layers[layer_index].weight)
        torch.nn.init.normal_(layers[layer_index].bias)
        return model
    else:
        return None


def rank_correlation(saliency, target):
    saliency = saliency.flatten()
    target = target.flatten()
    return stats.spearmanr(saliency, target)


def noabsscale(x):
    span = abs(np.percentile(x, 99))
    vmin = -span
    vmax = span
    return np.clip(x / (vmax - vmin), -1, 1)


def absscale(x):
    vmax = abs(np.percentile(x, 99))
    vmin = np.min(x)
    return np.clip((x - vmin) / (vmax - vmin), 0, 1)


def cascading_randomization(model):
    layers = extract_layers(model)
    layer_index = 0
    while layer_index < len(layers):
        result = layer_randomization(model, layer_index)
        if result is None:
            layer_index += 1
        else:
            layer_index += 1
            yield result


def independent_randomization(model):
    layers = extract_layers(model)
    model_copy = copy.deepcopy(model)
    for layer_index in range(len(layers)):
        model.load_state_dict(model_copy.state_dict())
        result = layer_randomization(model, layer_index)
        if result is None:
            layer_index += 1
        else:
            layer_index += 1
            yield result
