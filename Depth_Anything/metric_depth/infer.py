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
from .zoedepth.models.builder import build_model
from .zoedepth.utils.config import get_config


# FY = 256 * 0.6
# FX = 256 * 0.6


def process_images(model, dataset, fl=715.0873, max_depth = 100.0):
        
    input_dir = join(os.getcwd(), "Data/Run")
    output_dir = join(os.getcwd(), "Depth_Model/Results")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for image_path in tqdm(os.listdir(input_dir), desc="Processing Images"):

        idx = max([int(i.rsplit(".")[0]) for i in os.listdir(output_dir) if not "_" in i])
        try:
            color_image = Image.open(join(input_dir, image_path)).convert('RGB')
            shutil.copy(join(input_dir, image_path), join(output_dir, f"{idx+1}.png"))

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

            # Resize color image and depth to final size
            resized_pred = Image.fromarray(pred).resize((image_shape[1], image_shape[0]), Image.NEAREST)
            depth_map = np.array(resized_pred)

            # Visualize depth map
            norm_depth_map = np.clip(depth_map, 0, max_depth) 
            norm_depth_map = (norm_depth_map / max_depth) * 255
            norm_depth_map = norm_depth_map.astype(np.uint8)

            # Plot the depth map using a colormap for better visualization
            plt.imshow(norm_depth_map, cmap='plasma')
            plt.colorbar(label='m')
            plt.savefig(join(output_dir, str(idx+1)+"_any.png"))
            plt.close()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def main(model_name, pretrained_resource, dataset='nyu'):

    config = get_config(model_name, "eval", dataset)
    config.pretrained_resource = pretrained_resource
    
    model = build_model(config).to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    process_images(model, dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default='zoedepth', help="Name of the model to test")
    parser.add_argument("-p", "--pretrained_resource", type=str, default='local::./checkpoints/depth_anything_metric_depth_indoor.pt', help="Pretrained resource to use for fetching weights.")

    args = parser.parse_args()
    main(args.model, args.pretrained_resource)