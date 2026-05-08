import os.path
from data.base_dataset import BaseDataset, get_transform
import numpy as np
import torch
import glob
import random
import cv2



class AlignedNpzDataset(BaseDataset):


    def __init__(self, opt):
        BaseDataset.__init__(self, opt)


        dir_A_standard = os.path.join(opt.dataroot, opt.phase + '_A')
        dir_B_standard = os.path.join(opt.dataroot, opt.phase + '_B')


        dir_A_simple = os.path.join(opt.dataroot, 'A')
        dir_B_simple = os.path.join(opt.dataroot, 'B')


        if os.path.exists(dir_A_standard) and len(os.listdir(dir_A_standard)) > 0:
            self.dir_A = dir_A_standard
            self.dir_B = dir_B_standard
            print(f"📂 [Dataset] 这里的文件夹结构是标准的: {opt.phase}_A")
        elif os.path.exists(dir_A_simple):
            self.dir_A = dir_A_simple
            self.dir_B = dir_B_simple
            print(f"📂 [Dataset] 这里的文件夹结构是简化的: A/B")
        else:
            print(f"❌ [Error] 没找到有效的 A 文件夹！请检查路径: {opt.dataroot}")

            self.dir_A = dir_A_standard
            self.dir_B = dir_B_standard

        # ==========================================================

        # 获取文件列表
        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, '*.npz')))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, '*.npz')))

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

        print(f"📊 Dataset Info: A={self.A_size}, B={self.B_size}")
        if self.A_size == 0:
            raise ValueError(f"❌ 目录下没有找到 .npz 文件！路径: {self.dir_A}")

    def load_slice_from_npz(self, npz_path):

        try:
            data = np.load(npz_path)
            img = None

            if isinstance(data, np.lib.npyio.NpzFile):
                keys = list(data.files)
                candidate_keys = [k for k in keys if 'seg' not in k.lower() and 'mask' not in k.lower()]
                target_key = candidate_keys[0] if candidate_keys else keys[0]
                img = data[target_key]
            else:
                img = data

            if img.ndim == 3:
                depth = img.shape[0]
                img = img[depth // 2]

            img = img.astype(np.float32)
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
            img_uint8 = img_norm.astype(np.uint8)

            if img_uint8.shape != (256, 256):
                img_uint8 = cv2.resize(img_uint8, (256, 256))

            img_blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
            edge = cv2.Canny(img_blurred, 30, 100)

            img_tensor = torch.from_numpy(img_uint8).float()
            img_tensor = (img_tensor / 255.0 - 0.5) / 0.5
            img_tensor = img_tensor.unsqueeze(0)

            edge_tensor = torch.from_numpy(edge).float()
            edge_tensor = (edge_tensor / 255.0 - 0.5) / 0.5
            edge_tensor = edge_tensor.unsqueeze(0)

            final_tensor = torch.cat([img_tensor, edge_tensor], 0)
            return final_tensor

        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return torch.zeros((2, 256, 256)).float()

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]


        filename = os.path.basename(A_path)
        B_path = os.path.join(self.dir_B, filename)


        if not os.path.exists(B_path):
            B_path = self.B_paths[index % self.B_size]

        A_tensor = self.load_slice_from_npz(A_path)
        B_tensor = self.load_slice_from_npz(B_path)

        return {'A': A_tensor, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return self.A_size