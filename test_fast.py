import os
import argparse
import torch
import h5py
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from torch.nn import functional as F
import time

from unet.eff_unet import Eff_Unet
from utils.utils import calculate_metric_percase_dice

def save_prediction_with_reference(prediction, reference_path, save_dir, case_name):
    start = time.time()

    # 方向修正：先旋转再左右翻转，然后沿Z轴翻转
    for i in range(prediction.shape[0]):
        prediction[i] = np.fliplr(np.rot90(prediction[i], k=3))
    # prediction = prediction[::-1]  # 反转Z轴

    ref_img = sitk.ReadImage(reference_path)
    pred_img = sitk.GetImageFromArray(prediction.astype(np.uint8))
    pred_img.CopyInformation(ref_img)

    os.makedirs(save_dir, exist_ok=True)
    sitk.WriteImage(pred_img, os.path.join(save_dir, f"{case_name}_pred.nii.gz"))

    # print(f"🕒 保存预测耗时: {time.time() - start:.2f} 秒")

def test_single_volume(volume_path, model, patch_size, num_classes, save_path=None, ref_label_path=None, batch_size=4, device="cuda"):
    total_start = time.time()

    start = time.time()
    h5f = h5py.File(volume_path, 'r')
    image = h5f['image'][:]  # [D, H, W]
    label = h5f['label'][:]
    h5f.close()
    # print(f"🕒 数据加载耗时: {time.time() - start:.2f} 秒")

    prediction = np.zeros_like(label)
    model.eval()

    with torch.no_grad():
        start = time.time()
        slices = image[:, None, :, :]  # [D, 1, H, W]
        slices_tensor = torch.tensor(slices, dtype=torch.float32).to(device)
        resized = F.interpolate(slices_tensor, size=patch_size, mode='bilinear', align_corners=False)
        # print(f"🕒 图像resize耗时: {time.time() - start:.2f} 秒")

        start = time.time()
        pred_logits = model(resized)  # [D, num_classes, H, W]
        preds = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)  # [D, H, W]
        # print(f"🕒 模型推理耗时: {time.time() - start:.2f} 秒")

        start = time.time()
        preds_resized = F.interpolate(preds[:, None, :, :].float(), size=image.shape[1:], mode='nearest').squeeze(1)
        prediction = preds_resized.cpu().numpy().astype(np.uint8)
        # print(f"🕒 预测resize回原图耗时: {time.time() - start:.2f} 秒")

    start = time.time()
    metrics = []
    for i in range(1, num_classes):
        metric = calculate_metric_percase_dice((prediction == i), (label == i))
        metrics.append(metric)
    # print(f"🕒 指标计算耗时: {time.time() - start:.2f} 秒")

    if save_path and ref_label_path:
        save_prediction_with_reference(prediction, ref_label_path, save_path, os.path.basename(volume_path).split('.')[0])

    # print(f"🕒 总体耗时: {time.time() - total_start:.2f} 秒\n")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Either a .pth model path or a directory containing multiple .pth models.')
    parser.add_argument('--test_dir', type=str, default='data/Synapse/test_vol_h5')
    parser.add_argument('--save_pred', type=str, default=None)
    parser.add_argument('--ref_dir', type=str, default=None,
                        help='Directory containing reference .nii.gz labels for copying spacing/origin')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = sorted([f for f in os.listdir(args.test_dir) if f.endswith('.h5')])

    if os.path.isfile(args.model_dir) and args.model_dir.endswith('.pth'):
        # ---------- 单模型模式 ----------
        model_paths = [args.model_dir]
        is_multi = False
    elif os.path.isdir(args.model_dir):
        # ---------- 多模型模式 ----------
        model_paths = sorted([
            os.path.join(args.model_dir, f)
            for f in os.listdir(args.model_dir)
            if f.endswith('.pth')
        ])
        is_multi = True
    else:
        raise ValueError("Provided --model_dir must be a .pth file or a directory containing .pth files.")

    best_dice = -1.0
    best_model_path = None
    best_all_metrics = None

    for model_path in model_paths:
        print(f"\n🔧 正在加载模型: {model_path}")
        model = Eff_Unet(
            layers=[5, 5, 15, 10],
            embed_dims=[40, 80, 192, 384],
            downsamples=[True, True, True, True],
            num_classes=args.num_classes,
            fork_feat=True,
            vit_num=6
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        all_metrics = []

        print("🚀 开始测试...")
        for f in tqdm(test_files):
            f_path = os.path.join(args.test_dir, f)
            case_name = f.split('.')[0]

            ref_label_path = None
            if args.ref_dir:
                label_name = case_name.replace('img', 'label')
                ref_label_path = os.path.join(args.ref_dir, label_name + '.nii.gz')

            # 仅保存预测图像：1) 单模型模式 或 2) 是当前最佳模型（在最终一次循环外处理）
            save_pred_now = args.save_pred if (not is_multi) else None

            metrics = test_single_volume(
                f_path, model, (args.img_size, args.img_size),
                args.num_classes, save_pred_now, ref_label_path,
                batch_size=args.batch_size, device=device
            )
            all_metrics.append(metrics)

        all_metrics = np.array(all_metrics)
        avg = np.mean(all_metrics, axis=0)
        mean_dice = np.mean(avg[:, 0])
        print(f"\n📊 模型 {os.path.basename(model_path)} 平均 Dice: {mean_dice:.4f}")

        if not is_multi:
            # 单模型时直接输出
            print("===> Mean Dice / HD95 / JC / ASD by class (1 ~ N):")
            for i, cls in enumerate(range(1, args.num_classes)):
                print(f"Class {cls}: Dice={avg[i, 0]:.4f}, HD95={avg[i, 1]:.2f}, JC={avg[i, 2]:.4f}, ASD={avg[i, 3]:.2f}")
            print("===> Mean over all classes:")
            print(f"Dice: {np.mean(avg[:, 0]):.4f}, HD95: {np.mean(avg[:, 1]):.2f}, JC: {np.mean(avg[:, 2]):.4f}, ASD: {np.mean(avg[:, 3]):.2f}")
            return

        # 多模型时，记录最优模型信息
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_model_path = model_path
            best_all_metrics = all_metrics

    # 多模型最终：保存最优预测结果
    if is_multi and best_model_path and args.save_pred:
        print(f"\n✅ 保存 Dice 最佳模型 ({os.path.basename(best_model_path)}) 的预测结果...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        model.eval()
        os.makedirs(args.save_pred, exist_ok=True)

        for f in tqdm(test_files):
            f_path = os.path.join(args.test_dir, f)
            case_name = f.split('.')[0]
            ref_label_path = os.path.join(args.ref_dir, case_name.replace('img', 'label') + '.nii.gz') if args.ref_dir else None

            test_single_volume(
                f_path, model, (args.img_size, args.img_size),
                args.num_classes, args.save_pred, ref_label_path,
                batch_size=args.batch_size, device=device
            )

        avg = np.mean(best_all_metrics, axis=0)
        print("\n===> 🎯 Dice 最佳模型:", os.path.basename(best_model_path))
        print("===> Mean Dice / HD95 / JC / ASD by class (1 ~ N):")
        for i, cls in enumerate(range(1, args.num_classes)):
            print(f"Class {cls}: Dice={avg[i, 0]:.4f}, HD95={avg[i, 1]:.2f}, JC={avg[i, 2]:.4f}, ASD={avg[i, 3]:.2f}")

        print("===> Mean over all classes:")
        print(f"Dice: {np.mean(avg[:, 0]):.4f}, HD95: {np.mean(avg[:, 1]):.2f}, JC: {np.mean(avg[:, 2]):.4f}, ASD: {np.mean(avg[:, 3]):.2f}")


if __name__ == '__main__':
    main()
