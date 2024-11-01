import os, cv2, glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ToTensor(object):

    def __init__(self):
    
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # TODO
        self.resize = transforms.Resize((480, 640))

    def __call__(self, sample):

        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)
        image = self.normalize(image)
        image = self.resize(image)
        depth = self.to_tensor(depth)

        return {'image': image, 'depth': depth, 'dataset': "vkitti"}

    def to_tensor(self, pic):

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

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


class USOD10K(Dataset):

    def __init__(self, data_dir_root, do_kb_crop=True):
         
        self.image_files = glob.glob(os.path.join(
            data_dir_root, "RGB/*.png"))
        self.depth_files = [r.replace("RGB", "depth")
                            for r in self.image_files]
        
        self.do_kb_crop = True
        self.transform = ToTensor()

    def __getitem__(self, idx):

        image_path = self.image_files[idx]
        depth_path = self.depth_files[idx]

        image = Image.open(image_path)
        depth = Image.open(depth_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR |
                           cv2.IMREAD_ANYDEPTH)

        image = np.asarray(image, dtype=np.float32) / 255.0
        depth = np.asarray(depth, dtype=np.uint16) / 1.
        # depth = depth[..., None]

        sample = dict(image=image, depth=depth)
        sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_files)


def get_usod_loader(data_dir_root, batch_size=1, **kwargs):
    
    dataset = USOD10K(data_dir_root)
    return DataLoader(dataset, batch_size, **kwargs)
