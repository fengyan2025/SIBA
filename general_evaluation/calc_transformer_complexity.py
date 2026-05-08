import torch
from monai.networks.nets import SwinUNETR
from thop import profile

def main():
    print("🚀 正在初始化 SwinUNETR 模型...")
    netG = SwinUNETR(
        in_channels=1,
        out_channels=1,
        feature_size=24,
        spatial_dims=2
    )


    dummy_input = torch.randn(1, 1, 256, 256)

    print("⏳ 正在计算 Params 和 FLOPs...")
    macs, params = profile(netG, inputs=(dummy_input, ), verbose=False)

    params_m = params / 1e6
    flops_g = (macs * 2) / 1e9

    print("\n" + "="*40)
    print("✅ 纯 Transformer (SwinUNETR) 复杂度报告")
    print("="*40)
    print(f"📦 参数量 (Params): {params_m:.2f} M")
    print(f"💻 实际推理算力 (FLOPs): {flops_g:.2f} G")
    print("="*40)

if __name__ == '__main__':
    main()