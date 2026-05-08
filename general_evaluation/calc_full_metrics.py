import os
import glob
import numpy as np
import torch
import cv2
import shutil
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


results_dir = ''


gt_dir = ''


temp_root = ''
temp_real_dir = os.path.join(temp_root, 'real')
temp_fake_dir = os.path.join(temp_root, 'fake')


# ====================================================

def setup_dirs():
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_real_dir, exist_ok=True)
    os.makedirs(temp_fake_dir, exist_ok=True)


def read_npz_to_img(path):

    try:
        data = np.load(path)
        keys = list(data.files)

        valid_keys = [k for k in keys if 'seg' not in k and 'mask' not in k]
        key = valid_keys[0] if valid_keys else keys[0]
        img = data[key]


        if img.ndim == 3: img = img[img.shape[0] // 2]
        if img.ndim == 4: img = img[0, img.shape[1] // 2]


        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0
        img = img.astype(np.uint8)


        if img.shape != (256, 256):
            img = cv2.resize(img, (256, 256))
        return img
    except Exception as e:

        return None


def load_images_to_tensor(folder, device):

    paths = sorted(glob.glob(os.path.join(folder, '*.png')))


    batch = []
    for p in paths:
        img = cv2.imread(p)  # BGR
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
        t = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]
        batch.append(t)

    if not batch:
        return None
    return torch.stack(batch).to(device)  # uint8 tensor


def main():
    setup_dirs()
    print("🚀 启动全指标评估 (PSNR | SSIM | FID | IS)...")


    fake_paths = sorted(glob.glob(os.path.join(results_dir, '*fake_B.png')))
    print(f"📊 找到 {len(fake_paths)} 张生成结果，开始配对处理...")

    psnr_list = []
    ssim_list = []


    for fake_p in tqdm(fake_paths, desc="Calculating Pair Metrics"):

        img_fake = cv2.imread(fake_p, cv2.IMREAD_GRAYSCALE)


        filename = os.path.basename(fake_p)

        real_name = filename.replace('_fake_B.png', '.npz')
        real_p = os.path.join(gt_dir, real_name)

        if not os.path.exists(real_p):

            real_p_png = fake_p.replace('fake_B.png', 'real_B.png')
            if os.path.exists(real_p_png):
                img_real = cv2.imread(real_p_png, cv2.IMREAD_GRAYSCALE)
            else:
                continue
        else:
            img_real = read_npz_to_img(real_p)

        if img_fake is None or img_real is None: continue


        if img_fake.shape != img_real.shape:
            img_real = cv2.resize(img_real, (img_fake.shape[1], img_fake.shape[0]))


        p = psnr_func(img_real, img_fake, data_range=255)
        s = ssim_func(img_real, img_fake, data_range=255)
        psnr_list.append(p)
        ssim_list.append(s)



        shutil.copy(fake_p, os.path.join(temp_fake_dir, filename))

        cv2.imwrite(os.path.join(temp_real_dir, filename), img_real)


    avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0
    avg_ssim = sum(ssim_list) / len(ssim_list) if ssim_list else 0

    print("\n" + "=" * 40)
    print(f"✅ [1/2] 配对指标结果 (Paired Metrics):")
    print(f"   🏆 PSNR: {avg_psnr:.4f} dB  (理想 > 30)")
    print(f"   🏆 SSIM: {avg_ssim:.4f}     (理想 > 0.90)")
    print("=" * 40 + "\n")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⏳ [2/2] 正在计算分布指标 (FID & IS) on {device}...")


    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    is_metric = InceptionScore(normalize=False).to(device)

    fake_imgs = sorted(glob.glob(os.path.join(temp_fake_dir, '*.png')))
    real_imgs = sorted(glob.glob(os.path.join(temp_real_dir, '*.png')))

    batch_size = 50

    def get_batch(paths, start_idx, end_idx):
        batch_t = []
        for i in range(start_idx, end_idx):
            if i >= len(paths): break
            img = cv2.imread(paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            batch_t.append(t)
        if not batch_t: return None
        return torch.cat(batch_t).to(device)


    for i in tqdm(range(0, len(real_imgs), batch_size), desc="FID: Update Real"):
        batch = get_batch(real_imgs, i, i + batch_size)
        if batch is not None:
            fid_metric.update(batch, real=True)


    for i in tqdm(range(0, len(fake_imgs), batch_size), desc="FID & IS: Update Fake"):
        batch = get_batch(fake_imgs, i, i + batch_size)
        if batch is not None:
            fid_metric.update(batch, real=False)
            is_metric.update(batch)


    fid_score = fid_metric.compute().item()
    is_score, is_std = is_metric.compute()

    print("\n" + "=" * 40)
    print(f"✅ [2/2] 分布指标结果 (Distribution Metrics):")
    print(f"   🏆 FID: {fid_score:.4f}     (越低越好, < 20 优秀)")
    print(f"   🏆 IS:  {is_score.item():.4f} +/- {is_std.item():.4f} (越高越好)")
    print("=" * 40)


    shutil.rmtree(temp_root)
    print("\n🧹 临时文件已清理。测试结束！")


if __name__ == '__main__':
    main()