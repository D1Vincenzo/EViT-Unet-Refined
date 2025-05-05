import os
import random
import numpy as np
import nibabel as nib
import h5py
from time import time

# 原始训练集路径
img_dir = 'datasets/synapse_data/averaged-training-images'
label_dir = 'datasets/synapse_data/averaged-training-labels'

# 输出路径
train_npz_dir = 'data/Synapse/train_npz'
val_h5_dir = 'data/Synapse/test_vol_h5'
os.makedirs(train_npz_dir, exist_ok=True)
os.makedirs(val_h5_dir, exist_ok=True)

# 取所有 nii.gz 文件
all_cases = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])

# 随机打乱并划分为 80% 训练 / 20% 验证
random.seed(42)
random.shuffle(all_cases)
split_idx = int(0.8 * len(all_cases))
train_cases = all_cases[:split_idx]
val_cases = all_cases[split_idx:]

print(f"✅ 共 {len(all_cases)} 个样本，划分为：训练集 {len(train_cases)}，验证集 {len(val_cases)}")

# CT 值裁剪范围
upper = 275
lower = -125

start_time = time()

# 处理训练集（切片为 .npz）
for ct_file in train_cases:
    img_path = os.path.join(img_dir, ct_file)
    label_path = os.path.join(label_dir, ct_file.replace('_avg.nii.gz', '_avg_seg.nii.gz'))

    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    img = np.transpose(img, (2, 0, 1))
    label = np.transpose(label, (2, 0, 1))

    ct_number = ct_file.replace('.nii.gz', '')
    for s_idx in range(img.shape[0]):
        slice_img = img[s_idx]
        slice_lbl = label[s_idx]
        slice_no = f"{s_idx:03d}"
        out_name = f"{ct_number}_slice_{slice_no}"
        np.savez(os.path.join(train_npz_dir, out_name + ".npz"), image=slice_img, label=slice_lbl)

    print(f"📦 已保存训练切片：{ct_file}")

# 处理验证集（整 volume 为 .npy.h5）
for ct_file in val_cases:
    img_path = os.path.join(img_dir, ct_file)
    label_path = os.path.join(label_dir, ct_file.replace('_avg.nii.gz', '_avg_seg.nii.gz'))

    img = nib.load(img_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    img = np.clip(img, lower, upper)
    img = (img - lower) / (upper - lower)

    img = np.transpose(img, (2, 0, 1))
    label = np.transpose(label, (2, 0, 1))

    ct_number = ct_file.replace('.nii.gz', '')
    out_path = os.path.join(val_h5_dir, ct_number + '.npy.h5')
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('image', data=img)
        f.create_dataset('label', data=label)

    print(f"🧪 已保存验证体积：{ct_file}")

print(f"✅ 全部处理完成，用时 {(time() - start_time) / 60:.2f} 分钟")

# 路径配置
npz_dir = 'data/Synapse/train_npz'
h5_dir = 'data/Synapse/test_vol_h5'
list_dir = 'lists/lists_Synapse'
os.makedirs(list_dir, exist_ok=True)

# 获取所有切片名（不含扩展名）
train_list = [f.replace('.npz', '') for f in os.listdir(npz_dir) if f.endswith('.npz')]
train_list.sort()

# 获取所有验证 volume 名（去掉扩展名）
val_list = [f.replace('.npy.h5', '') for f in os.listdir(h5_dir) if f.endswith('.npy.h5')]
val_list.sort()

# 写入 train.txt
with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_list))
print(f'✅ 写入 train.txt，共 {len(train_list)} 条记录')

# 写入 test_vol.txt
with open(os.path.join(list_dir, 'test_vol.txt'), 'w') as f:
    f.write('\n'.join(val_list))
print(f'✅ 写入 test_vol.txt，共 {len(val_list)} 条记录')