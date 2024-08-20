import os, shutil
import argparse, os
import numpy as np
import torch
import matplotlib.pyplot as plt

from Depth_Anything.metric_depth.evaluate import custom_infer
from Depth_Anything.metric_depth.zoedepth.utils.arg_utils import parse_unknown
from image_segmentation import bkgrd_removal_pipeline
from estimate_parameters import depth_derivatives


def rename_img(img_name, f=False):

    img_path = os.path.join(os.getcwd(), "Data/Run", img_name)
    all_idx = os.listdir(os.path.join(os.getcwd(), 'Data/Archive'))

    if len(all_idx) > 0:
        max_len = max([len(i) for i in all_idx]) 
        max_idx = [i for i in all_idx if len(i)==max_len]
        save_idx = int(sorted(max_idx)[-1].rsplit('.')[0]) + 1
    
    else: 
        save_idx = 0

    new_path = os.path.join(os.getcwd(), "Data/Run", f"{save_idx}.png")
    copy_path = os.path.join(os.getcwd(), "Data/Archive", f"{save_idx}.png")

    print(f"New image name is: {save_idx}.png")

    os.rename(img_path, new_path)
    shutil.copy(new_path, copy_path)

    return f"{save_idx}.png"


def vis_one_img(file_name, res_path=None):

    if not res_path:
        res_path = os.path.join(os.getcwd(), "Data/Depth_Estim")
    
    idx = file_name.rsplit(".")[0]
    pt_idx = idx + ".pt"

    img_depth = torch.load(os.path.join(res_path, pt_idx)).cpu().numpy()
    img_depth = np.squeeze(img_depth)

    plt.imshow(img_depth, cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(res_path, idx+"_depth.png"))
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
        default="zoedepth", help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
        default="local::./Depth_Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt")
    parser.add_argument("-d", "--dataset", type=str, required=False,
        default='custom_outdoor')
    parser.add_argument("-viz_d", "--viz_depth", type=bool, required=False,
        default=True)
    
    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    images = sorted(os.listdir(os.getcwd() + "/Data/Run/"))
    image_new_ids = [rename_img(img_name) for img_name in images]

    depth_img_names = custom_infer(args.pretrained_resource, args.model, 
                 args.dataset, **overwrite_kwargs)
    
    real_dists = [1, 0.3, 0.3]

    for i, img_name in enumerate(image_new_ids):

        new_id = image_new_ids[i]
        new_img_path = os.getcwd() + "/Data/Archive/" + new_id
 
        if args.viz_depth:
            vis_one_img(new_id)

        # _, mask = bkgrd_removal_pipeline(new_img_path)

        depth_derivatives(img_name, image_new_ids[i], real_dist=real_dists[i], plot=False)

        os.remove(os.getcwd() + "/Data/Run/" + img_name)