import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
import torch
import glob


class PairedNpzDataset(BaseDataset):


    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')


        self.A_paths = sorted(glob.glob(os.path.join(self.dir_A, '*.npz')))
        self.B_paths = sorted(glob.glob(os.path.join(self.dir_B, '*.npz')))


        if len(self.A_paths) == 0:
            raise ValueError(f"❌ 在 {self.dir_A} 中没找到任何 .npz 文件！请检查路径。")
        if len(self.A_paths) != len(self.B_paths):
            raise ValueError(f"❌ A和B的文件数量不匹配! A: {len(self.A_paths)}, B: {len(self.B_paths)}")

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc

    def __getitem__(self, index):

        A_path = self.A_paths[index]
        B_path = self.B_paths[index]


        try:
            A_np = np.load(A_path)
            B_np = np.load(B_path)


            if isinstance(A_np, np.lib.npyio.NpzFile):
                A_img = A_np[A_np.files[0]]
            else:
                A_img = A_np

            if isinstance(B_np, np.lib.npyio.NpzFile):
                B_img = B_np[B_np.files[0]]
            else:
                B_img = B_np

        except Exception as e:
            print(f"Error loading {A_path}: {e}")
            raise e


        A_tensor = torch.from_numpy(A_img).float()
        B_tensor = torch.from_numpy(B_img).float()


        if A_tensor.ndim == 2:
            A_tensor = A_tensor.unsqueeze(0)
        if B_tensor.ndim == 2:
            B_tensor = B_tensor.unsqueeze(0)


        if A_tensor.max() > A_tensor.min():
            A_tensor = (A_tensor - A_tensor.min()) / (A_tensor.max() - A_tensor.min())
            A_tensor = (A_tensor - 0.5) / 0.5

        if B_tensor.max() > B_tensor.min():
            B_tensor = (B_tensor - B_tensor.min()) / (B_tensor.max() - B_tensor.min())
            B_tensor = (B_tensor - 0.5) / 0.5

        if A_tensor.shape[1] != 256 or A_tensor.shape[2] != 256:
            import torch.nn.functional as F
            A_tensor = F.interpolate(A_tensor.unsqueeze(0), size=(256, 256), mode='bilinear',
                                     align_corners=False).squeeze(0)
            B_tensor = F.interpolate(B_tensor.unsqueeze(0), size=(256, 256), mode='bilinear',
                                     align_corners=False).squeeze(0)

        return {'A': A_tensor, 'B': B_tensor, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)