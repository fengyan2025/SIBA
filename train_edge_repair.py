import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from edge_networks import SimpleEdgeUNet

class FastMRIEdgeDataset(Dataset):
    def __init__(self, root_dir):
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npz')]
        print(f"Dataset loaded: {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx])
        bad_stack = data['input']
        good_stack = data['target']


        slice_idx = np.random.randint(0, bad_stack.shape[0])

        bad_slice = bad_stack[slice_idx]
        good_slice = good_stack[slice_idx]


        bad_tensor = torch.from_numpy(bad_slice).float() / 255.0
        good_tensor = torch.from_numpy(good_slice).float() / 255.0

        return bad_tensor.unsqueeze(0), good_tensor.unsqueeze(0)



def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    os.makedirs(args.checkpoint_dir, exist_ok=True)


    dataset = FastMRIEdgeDataset(args.data_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


    model = SimpleEdgeUNet().to(device)


    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    criterion = nn.BCELoss()

    print("Start training Edge Repair Module...")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        pbar = tqdm(dataloader)
        for bad_img, good_img in pbar:
            bad_img = bad_img.to(device)
            good_img = good_img.to(device)

            preds = model(bad_img)


            loss = criterion(preds, good_img)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_description(f"Epoch {epoch + 1}/{args.epochs} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss / len(dataloader):.5f}")


        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(args.checkpoint_dir, f'edge_net_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved model to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='fastMRI edge pairs 路径')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=)
    parser.add_argument('--epochs', type=int, default=)  # 边缘修复很简单，20轮足够了
    parser.add_argument('--lr', type=float, default=)

    args = parser.parse_args()
    train(args)