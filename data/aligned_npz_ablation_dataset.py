import os.path
from data.base_dataset import BaseDataset
import numpy as np
import torch
import cv2



def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.npz'):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class AlignedNpzAblationDataset(BaseDataset):


    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, 'A')
        self.dir_B = os.path.join(opt.dataroot, 'B')
        if not os.path.exists(self.dir_A):
            self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')


        self.A_paths = sorted(make_dataset(self.dir_A, float("inf")))
        self.B_paths = sorted(make_dataset(self.dir_B, float("inf")))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.B_size]

        A_img = self.read_npz(A_path)
        B_img = self.read_npz(B_path)


        zero_edge = torch.zeros_like(A_img)
        final_A = torch.cat([A_img, zero_edge], 0)

        return {'A': final_A, 'B': B_img, 'A_paths': A_path, 'B_paths': B_path}

    def read_npz(self, path):
        try:
            data = np.load(path)
            keys = list(data.files)
            valid_keys = [k for k in keys if 'seg' not in k and 'mask' not in k]
            key = valid_keys[0] if valid_keys else keys[0]
            img = data[key]

            if img.ndim == 3: img = img[img.shape[0] // 2]
            if img.ndim == 4: img = img[0, img.shape[1] // 2]
            img = img.astype(np.float32)
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_norm = (img_norm - 0.5) / 0.5

            img_uint8 = ((img_norm * 0.5 + 0.5) * 255).astype(np.uint8)
            if img_uint8.shape != (256, 256):
                img_uint8 = cv2.resize(img_uint8, (256, 256))
                img_norm = (img_uint8 / 255.0 - 0.5) / 0.5

            return torch.from_numpy(img_norm).float().unsqueeze(0)
        except:
            return torch.zeros((1, 256, 256)).float()

    def __len__(self):
        return max(self.A_size, self.B_size)