import torch.nn as nn
import torchvision.models as tv_models


class TorchvisionClassifier(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        weights=None,
        with_transform: bool = True,
        with_categories: bool = True,
    ):
        super().__init__()
        self.model = model
        self.weights = weights
        self._transform = None
        self.categories = None

        if with_transform and weights is not None:
            self._transform = weights.transforms(antialias=True)
        if with_categories and weights is not None:
            self.categories = weights.meta.get("categories")

    def forward(self, x):
        # Return logits to align with torchvision default behavior.
        return self.model(x)

    def transform(self, x):
        if self._transform is None:
            raise ValueError("No weights configured, transform is unavailable")
        return self._transform(x)


MODEL_REGISTRY = {
    "vgg11": (tv_models.vgg11, tv_models.VGG11_Weights),
    "vgg13": (tv_models.vgg13, tv_models.VGG13_Weights),
    "vgg16": (tv_models.vgg16, tv_models.VGG16_Weights),
    "vgg19": (tv_models.vgg19, tv_models.VGG19_Weights),
    "resnet50": (tv_models.resnet50, tv_models.ResNet50_Weights),
    "resnet101": (tv_models.resnet101, tv_models.ResNet101_Weights),
    "resnet152": (tv_models.resnet152, tv_models.ResNet152_Weights),
    "inception_v3": (tv_models.inception_v3, tv_models.Inception_V3_Weights),
    "mobilenet_v3_small": (
        tv_models.mobilenet_v3_small,
        tv_models.MobileNet_V3_Small_Weights,
    ),
    "mobilenet_v3_large": (
        tv_models.mobilenet_v3_large,
        tv_models.MobileNet_V3_Large_Weights,
    ),
    "mobile_v3_small": (
        tv_models.mobilenet_v3_small,
        tv_models.MobileNet_V3_Small_Weights,
    ),
    "mobile_v3_large": (
        tv_models.mobilenet_v3_large,
        tv_models.MobileNet_V3_Large_Weights,
    ),
    "densenet121": (tv_models.densenet121, tv_models.DenseNet121_Weights),
    "densenet169": (tv_models.densenet169, tv_models.DenseNet169_Weights),
    "densenet201": (tv_models.densenet201, tv_models.DenseNet201_Weights),
}


def list_models():
    return sorted(MODEL_REGISTRY.keys())


def _resolve_weights(weights_enum, weights):
    if weights is None:
        return None
    if weights == "DEFAULT":
        return weights_enum.DEFAULT
    if isinstance(weights, str):
        if not hasattr(weights_enum, weights):
            raise ValueError(f"Unsupported weights '{weights}'")
        return getattr(weights_enum, weights)
    return weights


def _set_num_classes(model_name: str, model: nn.Module, num_classes: int):
    if model_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return
    if model_name.startswith("vgg"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return
    if model_name.startswith("inception"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return
    if model_name.startswith("mobilenet") or model_name.startswith("mobile"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return
    if model_name.startswith("densenet"):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return


def create_model(
    name: str,
    weights="DEFAULT",
    num_classes=None,
    device=None,
    eval_mode: bool = True,
    with_transform: bool = True,
    with_categories: bool = True,
    **kwargs,
):
    model_name = name.lower()
    if model_name not in MODEL_REGISTRY:
        options = ", ".join(list_models())
        raise ValueError(f"Unsupported model '{name}'. Available models: {options}")

    constructor, weights_enum = MODEL_REGISTRY[model_name]
    resolved_weights = _resolve_weights(weights_enum, weights)
    model = constructor(weights=resolved_weights, **kwargs)

    if num_classes is not None:
        _set_num_classes(model_name, model, num_classes)

    wrapped_model = TorchvisionClassifier(
        model,
        weights=resolved_weights,
        with_transform=with_transform,
        with_categories=with_categories,
    )

    if eval_mode:
        wrapped_model.eval()
    if device is not None:
        wrapped_model = wrapped_model.to(device)
    return wrapped_model
