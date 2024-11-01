import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):

    def __init__(self):

        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Or calculate the mean and std of your dataset and then run
        # self.normalize = transforms.Normalize(mean=[...], std=[...])

        self.normalize = lambda x : x
        self.resize = transforms.Resize((480, 640))

    def __call__(self, sample):

        image, depth = sample['image'], sample['depth'] 

        image = self.to_tensor(image)
        image = self.normalize(image)

        depth = np.expand_dims(depth, axis=2)  # TODO
        print(depth.shape)
        depth = self.to_tensor(depth)

        image = self.resize(image)
        depth = self.resize(depth)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class CUSTOM_Outdoor(Dataset):

    def __init__(self, data_dir_root):

        self.image_files = glob.glob(os.path.join(
            data_dir_root, 'Images/*.png'))
        self.depth_files = [r.replace("Images", "depth") 
            for r in self.image_files]

        self.Transform = ToTensor()

    def __getitem__(self, idx):
    
        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = np.asarray(Image.open(image_path), dtype=np.float32)  # change according to dataset's characteristics
        depth = np.asarray(Image.open(depth_path), dtype=np.float32)  # change according to dataset's characteristics

        image = image / 255.

        if image.shape[2] == 4:
            image = image[:, :, :3]
        
        return self.Transform(dict(image=image, depth=depth)) 
    
    def __len__(self):
        return len(self.image_files)
    

def custom_mean_std(path):

    dataset = CUSTOM_Outdoor(path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Initialize variables to store sums for each channel
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images_count = 0

    for imdep in loader:

        images = imdep['image'] / 255.  # or .float()

        batch_images_count = images.size(0)
        total_images_count += batch_images_count

        mean += images.mean([0, 2, 3]) * batch_images_count
        std += images.std([0, 2, 3]) * batch_images_count

    mean /= total_images_count
    std /= total_images_count

    print("Mean:", mean)  # [0.3965, 0.4515, 0.3851]
    print("Std:", std)   # [0.2398, 0.2236, 0.2589]


def get_custom_loader(data_dir_root, batch_size=1, num_workers=1, **kwargs):

    dataset = CUSTOM_Outdoor(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)