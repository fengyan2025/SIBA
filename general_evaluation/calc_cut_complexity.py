import torch
from thop import profile


from models import networks


def main():
    print("🚀 正在初始化 CUT (ResNet-9blocks) 生成器...")


    netG = networks.define_G(
        input_nc=1,
        output_nc=1,
        ngf=64,
        netG='resnet_9blocks',
        norm='instance',
        use_dropout=False,
        init_type='normal',
        init_gain=0.02,
        gpu_ids=[]
    )


    dummy_input = torch.randn(1, 1, 256, 256)

    print("⏳ 正在计算 Params 和 FLOPs...")
    macs, params = profile(netG, inputs=(dummy_input,), verbose=False)

    params_m = params / 1e6
    flops_g = (macs * 2) / 1e9

    print("\n" + "=" * 40)
    print("✅ 对比学习 GAN (CUT) 复杂度报告")
    print("=" * 40)
    print(f"📦 参数量 (Params): {params_m:.2f} M")
    print(f"💻 实际推理算力 (FLOPs): {flops_g:.2f} G")
    print("=" * 40)


if __name__ == '__main__':
    main()