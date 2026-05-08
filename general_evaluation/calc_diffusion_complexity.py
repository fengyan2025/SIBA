import torch
from diffusers import UNet2DModel
from thop import profile



class UNetWrapper(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, t):

        return self.unet(x, t, return_dict=False)[0]


def main():
    print("🚀 正在初始化 UNet 模型...")

    unet = UNet2DModel(
        sample_size=256,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


    total_params = sum(p.numel() for p in unet.parameters())
    params_m = total_params / 1e6


    wrapped_model = UNetWrapper(unet)
    dummy_input = torch.randn(1, 2, 256, 256)
    dummy_timestep = torch.tensor([10])

    print("⏳ 正在重新计算 FLOPs...")
    macs, _ = profile(wrapped_model, inputs=(dummy_input, dummy_timestep), verbose=False)
    flops_g = (macs * 2) / 1e9

    print("\n" + "=" * 40)
    print("✅ 扩散模型复杂度报告 (真实数据)")
    print("=" * 40)
    print(f"📦 参数量 (Params): {params_m:.2f} M")
    print(f"💻 单步算力 (1-Step FLOPs): {flops_g:.2f} G")

    inference_steps = 50
    total_flops = flops_g * inference_steps

    print("-" * 40)
    print(f"🔥 实际推理算力 (DDIM {inference_steps} 步): {total_flops:.2f} G")
    print("=" * 40)


if __name__ == '__main__':
    main()