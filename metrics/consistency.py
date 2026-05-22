import copy
import torch

from utils import visualize_absscale as absscale, visualize_noabsscale as noabsscale, rank_correlation


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
