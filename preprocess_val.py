import os
import random
import numpy as np
import nibabel as nib
import h5py
from time import time

# 路径配置
img_dir = 'datasets/RawData/Training/img'
label_dir = 'datasets/RawData/Training/label'
train_npz_dir = 'data/Synapse/train_npz'
val_npz_dir = 'data/Synapse/val_npz'
test_h5_dir = 'data/Synapse/test_vol_h5'
list_dir = 'lists/lists_Synapse'

os.makedirs(train_npz_dir, exist_ok=True)
os.makedirs(val_npz_dir, exist_ok=True)
os.makedirs(test_h5_dir, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

# 收集样本
all_cases = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])
random.seed(42)
random.shuffle(all_cases)

n = len(all_cases)
n_train = int(n * 0.8)
n_val = int(n * 0.1)
n_test = n - n_train - n_val

train_cases = all_cases[:n_train]
val_cases = all_cases[n_train:n_train + n_val]
test_cases = all_cases[n_train + n_val:]

print(f"✅ 共 {n} 个样本，划分为：训练集 {len(train_cases)}，验证集 {len(val_cases)}，测试集 {len(test_cases)}")

upper, lower = 275, -125
start_time = time()

def process_label(label):
    label[label > 8] = 0
    return label

def save_npz_slices(cases, target_dir, label_prefix):
    for ct_file in cases:
        img_path = os.path.join(img_dir, ct_file)
        label_path = os.path.join(label_dir, ct_file.replace('img', 'label'))

        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        label = process_label(label)

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
            np.savez(os.path.join(target_dir, out_name + ".npz"), image=slice_img, label=slice_lbl)

        print(f"📦 {label_prefix}切片完成：{ct_file}")

def save_h5_volumes(cases, target_dir):
    for ct_file in cases:
        img_path = os.path.join(img_dir, ct_file)
        label_path = os.path.join(label_dir, ct_file.replace('img', 'label'))

        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        label = process_label(label)

        img = np.clip(img, lower, upper)
        img = (img - lower) / (upper - lower)
        img = np.transpose(img, (2, 0, 1))
        label = np.transpose(label, (2, 0, 1))

        ct_number = ct_file.replace('.nii.gz', '')
        out_path = os.path.join(target_dir, ct_number + '.npy.h5')
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('image', data=img)
            f.create_dataset('label', data=label)

        print(f"🧪 测试体积完成：{ct_file}")

# 分别处理
save_npz_slices(train_cases, train_npz_dir, "训练")
save_npz_slices(val_cases, val_npz_dir, "验证")
save_h5_volumes(test_cases, test_h5_dir)

# 写入列表
train_list = sorted([f.replace('.npz', '') for f in os.listdir(train_npz_dir) if f.endswith('.npz')])
with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train_list))
print(f'✅ 写入 train.txt，共 {len(train_list)} 条')

val_list = sorted([f.replace('.npz', '') for f in os.listdir(val_npz_dir) if f.endswith('.npz')])
with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
    f.write('\n'.join(val_list))
print(f'✅ 写入 val.txt，共 {len(val_list)} 条')

test_list = sorted([f.replace('.npy.h5', '') for f in os.listdir(test_h5_dir) if f.endswith('.npy.h5')])
with open(os.path.join(list_dir, 'test_vol.txt'), 'w') as f:
    f.write('\n'.join(test_list))
print(f'✅ 写入 test_vol.txt，共 {len(test_list)} 条')

print(f"🎉 全部处理完成！耗时 {(time() - start_time):.2f} 秒")
