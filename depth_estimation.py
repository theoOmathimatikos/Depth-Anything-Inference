import argparse, os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

from Depth_Anything.metric_depth.evaluate import custom_infer
from Depth_Anything.metric_depth.infer import main as mn
from Depth_Anything.metric_depth.zoedepth.utils.arg_utils import parse_unknown
from Depth_Model.UDepth.inference import run_code


def visualize_dir_depth(img_names):

    for file_name in img_names:
        vis_one_img(file_name)


def vis_one_img(file_name, res_path=None):

    if not res_path:
        res_path = os.path.join(os.getcwd(), "Data/Results")
    
    idx = file_name.rsplit(".")[0]
    pt_idx = idx + ".pt"

    img_depth = torch.load(os.path.join(res_path, pt_idx)).cpu().numpy()
    img_depth = np.squeeze(img_depth)

    plt.imshow(img_depth, cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(res_path, idx+"_depth_cm.png"))
    plt.close()


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", type=str,
    #     default="zoedepth", help="Name of the model to evaluate")
    # parser.add_argument("-p", "--pretrained_resource", type=str,
    #     default="local::./Depth_Anything/metric_depth/checkpoints/depth_anything_metric_depth_outdoor.pt")
    # parser.add_argument("-d", "--dataset", type=str, required=False,
    #     default='custom_outdoor')
        
    # args, unknown_args = parser.parse_known_args()
    # overwrite_kwargs = parse_unknown(unknown_args)
    
    # mn(args.model, args.pretrained_resource)

    run_code(use_rmi=False)
