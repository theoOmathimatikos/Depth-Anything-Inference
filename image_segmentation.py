# https://huggingface.co/briaai/RMBG-1.4
# pip install -qr https://huggingface.co/briaai/RMBG-1.4/resolve/main/requirements.txt

# Also try: https://huggingface.co/facebook/mask2former-swin-large-coco-panoptic

import os
import torch
import matplotlib.pyplot as plt
from transformers import pipeline
import numpy as np


# Remove background
def bkgrd_removal_pipeline(img_path, fd="Data"):

    img_idx = img_path.rsplit("/")[-1].rsplit(".")[0]

    dev = 1 if torch.cuda.is_available() else -1
    pipe = pipeline("image-segmentation", 
                    model="briaai/RMBG-1.4", 
                    trust_remote_code=True)  # device=dev
    
    # The model returns a soft mask: a value from 0 to 254, 
    # representing the confidence that the pixel belongs to the bckgrd
    pillow_mask = pipe(img_path, return_mask = True)
    pillow_image = pipe(img_path)

    pillow_mask_np = np.array(pillow_mask)
    pillow_image_np = np.array(pillow_image)

    # Construct a hard mask
    mask = pillow_mask_np>=127
    mask = mask.astype(int)

    plt.imshow(pillow_mask)
    plt.savefig(os.getcwd() + f"/{fd}/Remove_Bckgrd/" + img_idx + "_img.png")

    plt.imshow(pillow_mask)
    plt.savefig(os.getcwd() + f"/{fd}/Remove_Bckgrd/" + img_idx + "_mask.png")

    return pillow_image_np, mask


def run_all_images(folder_path, fd):

    for img_path in os.listdir(folder_path):
        bkgrd_removal_pipeline(os.path.join(folder_path, img_path), fd)


# Panoptic Segmentation
# TODO


if __name__ == "__main__":

    # run_all_images(os.getcwd() + "/Reports/Images/", "Reports")

    bkgrd_removal_pipeline(os.getcwd() + "/Reports/Images/3.png")