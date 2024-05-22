import sys

sys.path.append(".")
import numpy as np
import os
from metrics.consistency import noabsscale, absscale, rank_correlation


def get_correlation(saliency, target, scale="abs"):
    if scale == "abs":
        saliency = absscale(saliency)
        target = absscale(target)
    elif scale == "noabs":
        saliency = noabsscale(saliency)
        target = noabsscale(target)
    else:
        raise ValueError("Scale must be abs or noabs")
    return rank_correlation(saliency, target).correlation


def main(dataset, model, scale):
    datapath = f"./saved_data/{model}/{dataset}"
    explainers = os.listdir(datapath + "Normal/")
    explainers.sort()
    for explainer in explainers:
        correlations_random = []
        correlations_bias = []
        imgs = os.listdir(datapath + "Normal/" + explainer)
        for img in imgs:
            saliency = np.load(datapath + "Normal/" + explainer + "/" + img)
            saliency = np.squeeze(saliency)
            target_random = np.load(datapath + "Random/" + explainer + "/" + img)
            target_random = np.squeeze(target_random)
            target_bias = np.load(datapath + "Bias/" + explainer + "/" + img)
            target_bias = np.squeeze(target_bias)
            correlations_random.append(get_correlation(saliency, target_random, scale))
            correlations_bias.append(get_correlation(saliency, target_bias, scale))
        print(explainer, scale)
        correlations_random = np.array(correlations_random)
        correlations_bias = np.array(correlations_bias)
        print(
            "Random: ",
            np.mean(np.abs(correlations_random[~np.isnan(correlations_random)])),
        )
        print(
            "Bias: ", np.mean(np.abs(correlations_bias[~np.isnan(correlations_bias)]))
        )
        if not os.path.exists("./results/" + dataset + "/" + model):
            os.makedirs("./results/" + dataset + "/" + model)
        np.save(
            "./results/"
            + dataset
            + "/"
            + model
            + "/"
            + explainer
            + "_"
            + scale
            + "_random.npy",
            correlations_random,
        )
        np.save(
            "./results/"
            + dataset
            + "/"
            + model
            + "/"
            + explainer
            + "_"
            + scale
            + "_bias.npy",
            correlations_bias,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Get correlation")
    parser.add_argument("-s", "--scale", type=str, default="noabs")
    parser.add_argument("-d", "--dataset", type=str, default="mnist")
    parser.add_argument("-m", "--model", type=str, default="MLP")
    args = parser.parse_args()
    main(args.dataset, args.model, args.scale)
