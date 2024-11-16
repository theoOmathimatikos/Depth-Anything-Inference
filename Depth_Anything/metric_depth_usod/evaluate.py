# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat
import os
import argparse
from pprint import pprint

import torch
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from .zoedepth.utils.easydict import EasyDict as edict
from .zoedepth.data.data_mono import DepthDataLoader
from .zoedepth.models.builder import build_model
from .zoedepth.utils.arg_utils import parse_unknown
from .zoedepth.utils.config import change_dataset, get_config, ALL_EVAL_DATASETS, ALL_INDOOR, ALL_OUTDOOR
from .zoedepth.utils.misc import (RunningAverageDict, colors, compute_metrics,
                        count_parameters)
from .zoedepth.utils.misc import colorize



@torch.no_grad()
def infer(model, images, **kwargs):
    """Inference with flip augmentation"""
    # images.shape = N, C, H, W
    def get_depth_from_prediction(pred):
        if isinstance(pred, torch.Tensor):
            pred = pred  # pass
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        elif isinstance(pred, dict):
            pred = pred['metric_depth'] if 'metric_depth' in pred else pred['out']
        else:
            raise NotImplementedError(f"Unknown output type {type(pred)}")
        return pred

    pred1 = model(images, **kwargs)
    pred1 = get_depth_from_prediction(pred1)

    pred2 = model(torch.flip(images, [3]), **kwargs)
    pred2 = get_depth_from_prediction(pred2)
    pred2 = torch.flip(pred2, [3])

    mean_pred = 0.5 * (pred1 + pred2)

    return mean_pred


@torch.no_grad()
def evaluate(model, test_loader, config, round_vals=True, round_precision=3):
    
    model.eval()
    metrics = RunningAverageDict()

    for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        if 'has_valid_depth' in sample:
            if not sample['has_valid_depth']:
                continue
        
        image, depth = sample['image'], sample['depth']
        image, depth = image.cuda(), depth.cuda()
        depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
        focal = sample.get('focal', torch.Tensor(
            [715.0873]).cuda())  # This magic number (focal) is only used for evaluating BTS model
        pred = infer(model, image, dataset=config.dataset, focal=focal)

        # Save image, depth, pred for visualization
        if "save_images" in config and config.save_images:

            os.makedirs(config.save_images, exist_ok=True)
            d = colorize(depth.squeeze().cpu().numpy(), 0, 10)
            p = colorize(pred.squeeze().cpu().numpy(), 0, 10)

            im = transforms.ToPILImage()(image.squeeze().cpu())
            im.save(os.path.join(config.save_images, f"{i}_img.png"))
            Image.fromarray(d).save(os.path.join(config.save_images, f"{i}_depth.png"))
            Image.fromarray(p).save(os.path.join(config.save_images, f"{i}_pred.png"))

        metrics.update(compute_metrics(depth, pred, config=config))

    if round_vals:
        def r(m): return round(m, round_precision)
    else:
        def r(m): return m
    metrics = {k: r(v) for k, v in metrics.get_value().items()}
    return metrics


def main(config):

    model = build_model(config)
    model = model.cuda()

    test_loader = DepthDataLoader(config, 'online_eval').data
    metrics = evaluate(model, test_loader, config)

    print(f"{colors.fg.green}")
    print(metrics)
    print(f"{colors.reset}")
    
    metrics['#params'] = f"{round(count_parameters(model, include_all=True)/1e6, 2)}M"
    return metrics

"""
@torch.no_grad()
def custom_infer(pretrained_resource, model_name, dataset, **kwargs):

    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)

    model = build_model(config)
    model = model.cuda().eval()

    test_loader = DepthDataLoader(config, "test").data

    collect_names = []
    for sample in tqdm(test_loader, total=len(test_loader)):

        image = sample['image'].cuda()
        rt, pth = sample['route'], sample['path'][0]
        name = rt[0]+"/Data/Depth_Estim/"+pth

        try:

            pred = infer(model, image, dataset=sample['dataset'][0])
            torch.save(pred, name.rsplit(".", 1)[0]+".pt")
            collect_names.append(pth)
             
            img_pth = name.rsplit(".", 1)[0]
            p = colorize(pred.squeeze().cpu().numpy(), 0, 10)
            Image.fromarray(p).save(img_pth+".png")

            # fig, ax = plt.subplots()
            # cax = ax.imshow(p, cmap="inferno")
            # fig.colorbar(cax)
            # plt.savefig(img_pth+"_bar.png", bbox_inches='tight')

        except RuntimeError:

            print("An error occured")

    return collect_names
"""

def eval_model(model_name, pretrained_resource, dataset='nyu', **kwargs):

    # Load default pretrained resource defined in config if not set
    overwrite = {**kwargs, "pretrained_resource": pretrained_resource} if pretrained_resource else kwargs
    config = get_config(model_name, "eval", dataset, **overwrite)
    # config = change_dataset(config, dataset)  # change the dataset
    # pprint(config)
    print(f"Evaluating {model_name} on {dataset}...")
    metrics = main(config)
    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str,
                        required=False, default= "zoedepth", help="Name of the model to evaluate")
    parser.add_argument("-p", "--pretrained_resource", type=str,
                        required=False, default="local::./checkpoints/depth_anything_metric_depth_outdoor.pt",
                        help="Pretrained resource to use for fetching weights. If not set, default resource from model config is used,  Refer models.model_io.load_state_from_resource for more details.")
    parser.add_argument("-d", "--dataset", type=str, required=False,
                        default='nyu', help="Dataset to evaluate on")

    args, unknown_args = parser.parse_known_args()
    overwrite_kwargs = parse_unknown(unknown_args)

    # if "ALL_INDOOR" in args.dataset:
    #     datasets = ALL_INDOOR
    # elif "ALL_OUTDOOR" in args.dataset:
    #     datasets = ALL_OUTDOOR
    # elif "ALL" in args.dataset:
    #     datasets = ALL_EVAL_DATASETS
    # elif "," in args.dataset:
    #     datasets = args.dataset.split(",")
    # else:
    #     datasets = [args.dataset]
    #
    # for dataset in datasets:
    #     eval_model(args.model, pretrained_resource=args.pretrained_resource,
    #                 dataset=dataset, **overwrite_kwargs)

    # custom_infer(args.pretrained_resource, args.model, args.dataset, **overwrite_kwargs)

    dataset = "custom_outdoor"
    _ = eval_model(args.model, pretrained_resource=args.pretrained_resource,
                    dataset=dataset, **overwrite_kwargs)
    
