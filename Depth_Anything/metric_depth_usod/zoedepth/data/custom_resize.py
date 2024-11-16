import os
from PIL import Image


def crop_and_resize(image_path, depth_image_path, output_path, depth_output_path, target_size=(640, 560)):
    """
    Crop and resize an image to the target size (h, w) by cropping the center.
    
    :param image_path: Path to the input image
    :param depth_image_path: Path to the relative depth image
    :param output_path: Path to save the resized image
    :param depth_output_path: Path to save the relative depth image
    :param target_size: Tuple of (height, width) for the desired size
    """
    
    # Open the image and get the original dimensions
    img = Image.open(image_path)
    depth_img = Image.open(depth_image_path)
    
    width, height = img.size
    target_h, target_w = target_size
    
    # Determine the amount to crop from each side
    crop_top, crop_left = (height - target_h) // 2, (width - target_w) // 2
    crop_bottom, crop_right = crop_top + target_h, crop_left + target_w
    
    # Perform center cropping
    img_cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    depth_img_cropped = depth_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    # Resize to ensure it fits exactly the target size
    img_resized = img_cropped.resize(target_size, Image.ANTIALIAS)
    depth_img_resized = depth_img_cropped.resize(target_size, Image.ANTIALIAS)
    
    # Save the result
    img_resized.save(output_path)
    depth_img_resized.save(depth_output_path)


def resize_custom_dataset():

    pth = os.path.join(os.getcwd(), "Depth_Anything/metric_depth/custom_dataset")

    input_dir = os.path.join(pth, "Images")  
    depth_input_dir = os.path.join(pth, "depth")

    output_dir = os.path.join(pth, "Resized_Images")
    depth_output_dir = os.path.join(pth, "Resized_depth")

    for filename in os.listdir(input_dir):

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        depth_input_path = os.path.join(depth_input_dir, filename)
        depth_output_path = os.path.join(depth_output_dir, filename)

        crop_and_resize(input_path, depth_input_path, output_path, 
                        depth_output_path)


def get_custom_dataset_size():

    dir = os.path.join(os.getcwd(), "Depth_Anything/metric_depth/custom_dataset/Images")

    for filename in os.listdir(dir):

        img = Image.open(os.path.join(dir, filename))
        print(f"Image: {filename}, shape: {img.size}")


if __name__ == "__main__":

    # get_custom_dataset_size()
    resize_custom_dataset()