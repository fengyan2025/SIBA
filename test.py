import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import glob
import shutil

if __name__ == '__main__':
    opt = TestOptions().parse()


    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1


    dataset = create_dataset(opt)


    model = create_model(opt)
    model.setup(opt)
    model.eval()


    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s' % opt.name)

    clean_dir = os.path.join(web_dir, 'images_clean')
    os.makedirs(clean_dir, exist_ok=True)

    print(f"🚀 开始测试 (共 {len(dataset)} 张切片)...")

    for i, data in enumerate(dataset):
        if i >= opt.num_test and opt.num_test > 0:
            break


        img_path = data['A_paths'][0]
        short_name = os.path.basename(img_path).replace('.npz', '')


        model.set_input(data)
        model.test()


        visuals = model.get_current_visuals()


        if not opt.no_save:
            save_images(webpage, visuals, [short_name], width=opt.display_winsize)

    webpage.save()


    print("📦 正在整理最终结果...")
    source_pattern = os.path.join(web_dir, 'images', '*fake_B.png')
    for filepath in glob.glob(source_pattern):
        filename = os.path.basename(filepath)
        shutil.copy(filepath, os.path.join(clean_dir, filename))

    print(f"✅ 测试完成！")
    print(f"📂 生成结果已整理在: {clean_dir}")
    print(f"💡 下一步: 请运行 final_score_fixed.py 计算指标。")