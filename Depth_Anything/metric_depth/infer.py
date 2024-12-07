import argparse
import os, shutil
from os.path import dirname as dr, join 
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.easydict import EasyDict as edict



# FY = 256 * 0.6
# FX = 256 * 0.6


def process_images(model, dataset, in_dir, out_dir,
                   out_end, fl=30.0, max_depth = 40.0):
        
    input_dir = join(os.getcwd(), in_dir)
    output_dir = join(os.getcwd(), out_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for image_path in tqdm(os.listdir(input_dir)):

        idx = image_path.rsplit(".")[0]
        # idx = max([int(i.rsplit(".")[0]) for i in os.listdir(output_dir) if not "_" in i])
        
        color_image = Image.open(join(input_dir, image_path)).convert('RGB')

        image_tensor = transforms.ToTensor()(color_image)
        image_shape = image_tensor.shape[1:]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Model depth prediction
        pred = model(image_tensor, dataset=dataset, focal=fl)

        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]
        
        pred = pred.squeeze().detach().cpu().numpy()

        # # Resize color image and depth to final size
        # if save_as_png:

        #     pred = torch.tensor(pred) + torch.normal(0, 0.1, size=pred.shape)
        #     pred = pred.numpy()
        #     pred = np.clip(pred, 0, max_depth) 
        #     pred = pred.astype(np.uint8)
        #     resized_pred = Image.fromarray(pred).resize((image_shape[1], image_shape[0]), Image.NEAREST)
        #     resized_pred.save(join(output_dir, image_path))
            
        resized_pred = Image.fromarray(pred).resize((image_shape[1], image_shape[0]), Image.NEAREST)
        depth_map = np.array(resized_pred)
        # torch.save(torch.tensor(depth_map), join(output_dir, idx+out_end+".pt"))

        # Clip and round values
        norm_depth_map = np.clip(depth_map, 0, max_depth) 
        norm_depth_map = norm_depth_map.astype(np.uint8)

        # Plot the depth map using a colormap for better visualization
        plt.imshow(norm_depth_map, cmap='plasma')
        plt.colorbar(label='(m)')
        plt.savefig(join(output_dir, idx+out_end+".png"))
        plt.close()


def main(model_name, pretrained_resource, in_dir, 
         out_dir, out_end, dataset='nyu'):

    config = get_config(model_name, "eval", dataset)
    config.pretrained_resource = pretrained_resource
    
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    process_images(model, dataset, in_dir, out_dir, out_end)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    # parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_outdoor.pt', help="Pretrained resource to use for fetching weights.")
    # parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./train_checkpoint/epoch_1.pt', help="Pretrained resource to use for fetching weights.")

    # args = parser.parse_args()

    args = edict({"dataset": "Depth_Anything/metric_depth/custom_data/Images", 
            "output": "Depth_Anything/metric_depth/custom_data/depth",
            "pretrained_resource": 'local::/Depth_Anything/checkpoints/depth_anything_metric_depth_outdoor.pt',
            # "pretrained_resource": 'local::/Depth_Anything/checkpoints/latest_checkpoint.pth',
            "model": "zoedepth"})

    
    main(args.model, args.pretrained_resource, in_dir="Depth_Anything/metric_depth/custom_data/Images",
         out_dir="Depth_Anything/metric_depth/custom_data/depth", out_end="_base")
    