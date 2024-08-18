import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ToTensor(object):

    def __init__(self):
        ### TODO. Check results from other transformations
        # self.normalize = transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.normalize = lambda x : x
        # self.resize = transforms.Resize((224, 224))

    def __call__(self, sample):

        image = sample['image']
        image = self.to_tensor(image)
        image = self.normalize(image)
        # image = self.resize(image)

        return {'image': image, 'dataset': "custom_outdoor", 'route': sample['route'], 'path': sample['path']}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        img = torch.ByteTensor(
            torch.ByteStorage.from_buffer(pic.tobytes()))
        nchannel = 3

        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img


class CUSTOM_Outdoor(Dataset):

    def __init__(self, data_dir_root):

        self.data_path = data_dir_root
        self.image_files = glob.glob(os.path.join(
            self.data_path, '*.png'
        ))
        self.Transform = ToTensor()

    def __getitem__(self, idx):
    
        image_path = self.image_files[idx]
        image = np.asarray(Image.open(image_path), dtype=np.float32) / 255.0

        if image.shape[2] == 4:
            image = image[:, :, :3]

        sample = dict(image=image, dataset="custom_outdoor", 
                      route=os.getcwd(), 
                      path=image_path.rsplit("/")[-1])

        # return sample
        return self.Transform(sample)
    
    def __len__(self):
        return len(self.image_files)


def get_custom_loader(data_dir_root, batch_size=1, num_workers=1, **kwargs):

    dataset = CUSTOM_Outdoor(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)