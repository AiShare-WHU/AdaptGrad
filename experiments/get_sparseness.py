import sys

sys.path.append(".")
import numpy as np
import os

from metrics.sparseness import gini

models = [
    # "vgg16",
    # "inception_v3",
    "mobile_v3_small",
]


def main(model):
    datapath = f"./saved_data/{model}/imagenet/"
    explainers = os.listdir(datapath)
    explainers.sort()
    for explainer in explainers:
        ginis = []
        imgs = os.listdir(datapath + explainer)
        for img in imgs:
            saliency = np.load(datapath + explainer + "/" + img)
            saliency = saliency.flatten()
            ginis.append(gini(saliency))
        ginis = np.array(ginis)
        print(model, explainer)
        print("Gini: ", np.mean(ginis))
        if not os.path.exists("./results/imagenet/" + model):
            os.makedirs("./results/imagenet/" + model)
        np.save("./results/imagenet/" + model + "/" + explainer + "_gini.npy", ginis)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="vgg16")
    args = parser.parse_args()
    main(args.model)
