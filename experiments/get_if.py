import sys

from matplotlib import pyplot as plt

sys.path.append(".")
import numpy as np
import os
import joblib
import torch as tc
import torchvision as tv
from tqdm import tqdm
from metrics.pic import (
    generate_random_mask,
    compute_pic_metric,
    aggregate_individual_pic_results,
    ComputePicMetricError,
)
from utils import (
    array_to_tensor_imagenet,
    tensor_to_array_imagenet,
    model_call,
    salience_to_array,
)
from models import create_model


def set_seed(seed: int):
    np.random.seed(seed)
    tc.manual_seed(seed)
    if tc.cuda.is_available():
        tc.cuda.manual_seed_all(seed)
    tc.backends.cudnn.deterministic = True
    tc.backends.cudnn.benchmark = False


def main(model_name, seed=0):
    set_seed(seed)
    if tc.cuda.is_available():
        print("CUDA is available")
        device = tc.device("cuda:0")
    else:
        print("CUDA is not available")
        device = tc.device("cpu")
    datapath = f"./saved_data/{model_name}/imagenet/"
    imgpath = "./data/imagenet/images/"
    model = create_model(model_name, device=device, eval_mode=True)
    explainers = os.listdir(datapath)
    explainers.sort()
    explainers = [explainer for explainer in explainers if "Integrated" in explainer]
    imgs = []
    pic_results = {}
    for explainer in explainers:
        pic_results[explainer] = []
        eimgs = os.listdir(datapath + explainer)
        if len(eimgs) > len(imgs):
            imgs = eimgs
    imgs.sort()
    for i, img_id in enumerate(imgs):
        img_input = tv.io.read_image(imgpath + img_id.split(".")[0] + ".JPEG").to(
            device
        )
        img_input = model.transform(img_input)
        img_input = img_input.unsqueeze(0)
        for explainer in explainers:
            saliency = np.load(datapath + explainer + "/" + img_id)
            array = tensor_to_array_imagenet(img_input[0])
            func = model_call(model, device=device)
            for j in range(10):
                random_mask = generate_random_mask(
                    img_input.shape[2], img_input.shape[3]
                )
                try:
                    result = compute_pic_metric(
                        array,
                        saliency,
                        random_mask,
                        func,
                        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                    )
                    pic_results[explainer].append(result)
                except ComputePicMetricError as e:
                    continue
            if i % 10 == 0:
                if len(pic_results[explainer]) == 0 or not pic_results[explainer]:
                    continue
                sample_result = aggregate_individual_pic_results(pic_results[explainer])
                print(f"{i/len(imgs)*100:.2f}", explainer, sample_result.auc)

    for explainer in explainers:
        all_result = aggregate_individual_pic_results(pic_results[explainer])
        print(explainer, all_result.auc)
        if not os.path.exists("./results/imagenet/" + model_name):
            os.makedirs("./results/imagenet/" + model_name)
        joblib.dump(all_result, f"./results/imagenet/{model_name}/{explainer}_pic.pkl")
        # np.save("./results/imagenet/" + model_name + "/" + explainer + "_pic.npy", all_result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="vgg16")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.model, args.seed)
