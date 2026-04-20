# AdaptGrad
Official Implementation of paper “AdaptGrad: Adaptive Sampling to Reduce Noise”, accepted in NIPS 2025.

![examples](./figs/vis_vgg.jpg)

## Requirements

To install requirements:
    
```shell
pip install -r requirements.txt
```

## Downloading the datasets

MNIST are available in torchvision. For ImageNet, please download the dataset from [here](http://www.image-net.org/challenges/LSVRC/2012/).

For ImageNet, the dataset should be placed in the `./data/imagenet/images/` folder. And the labels should be placed in the `./data/imagenet/labels/` folder.

## Pre-trained Models

You can find the pre-trained models in the `saved_models` folder.

For ImageNet models, we use torchvision weights through a unified model factory.
Supported model names include:

- `vgg11`, `vgg13`, `vgg16`, `vgg19`
- `resnet50`, `resnet101`, `resnet152`
- `inception_v3`
- `mobilenet_v3_small`, `mobilenet_v3_large`
- `densenet121`, `densenet169`, `densenet201`

Example:

```python
from models import create_model

model = create_model("resnet50")
img = model.transform(img)
logits = model(img.unsqueeze(0))
probs = logits.softmax(dim=1)
```

## Results

All the results will be saved in the `results` folder.

## Examples

See `vis_imagenet.ipynb` for examples of how to use the code.

## Experiments

To reproduce the experiments in the paper, please run the following commands:

### Consistency

#### MNIST

Get the salience of the MNIST dataset:
```shell
python experiments/get_salience_mnist.py with dataset=MNIST model_name=MLP kind=Normal
```

Get the salience of the MNIST dataset with random labels:
```shell
python experiments/get_salience_mnist.py with dataset=MNIST model_name=MLP kind=Random
```

### Invariance

#### MNIST

Get the salience of the MNIST dataset with bias shift:
```shell
python experiments/get_salience_mnist.py with dataset=MNIST model_name=MLP kind=Bias t=0.5 device_id=0
```

### Get the Consistency and Invariance

Get the Consistency and Invariance of the MNIST dataset (Regularization with noabs):
```shell
python experiments/get_correlation.py -s noabs -d MNIST -m MLP
```

Get the Consistency and Invariance of the MNIST dataset (Regularization with abs):
```shell
python experiments/get_correlation.py -s abs -d MNIST -m MLP
```

### Sparseness and Information Level

Get the saliency map first:
```shell
python experiments/get_salience_imagenet.py -m vgg16 # option: inception_v3, resnet50, densenet121, mobilenet_v3_small
```

Get the Sparseness of the VGG16 model on the ImageNet dataset:
```shell
python experiments/get_sparseness.py -m vgg16
```


Get the Information Level of the VGG16 model on the ImageNet dataset:
```shell
python experiments/get_if.py -m vgg16
```
