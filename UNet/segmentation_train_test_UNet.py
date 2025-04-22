# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from glob import glob
import numpy as np
import tensorflow as tf
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import random
from sklearn.metrics import jaccard_score

# ✅ 데이터 로딩 함수
def load_nc_data(nc_paths, selected_vars, label_keys=['label_0_ar', 'LABELS'], normalize=True,
                 target_size=(768, 1152)):
    input_list = []
    label_list = []

    for path in nc_paths:
        print(f"🔄 Loading file: {path}")
        try:
            nc = Dataset(path, 'r')

            data_group = nc.groups['data'] if 'data' in nc.groups else nc
            label_group = nc.groups['labels'] if 'labels' in nc.groups else nc

            actual_label_key = None
            for key in label_keys:
                if key in label_group.variables:
                    actual_label_key = key
                    break
            if actual_label_key is None:
                raise KeyError("No valid label variable found")

            inputs = []
            for var in selected_vars:
                data = data_group.variables[var][:]
                if data.ndim == 3:
                    data = data[0]
                inputs.append(data)
            inputs = np.stack(inputs, axis=-1)
            inputs = np.nan_to_num(inputs, nan=0.0)
            if normalize:
                inputs = (inputs - np.min(inputs)) / (np.max(inputs) - np.min(inputs))
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
            inputs = tf.image.resize(inputs, target_size)

            label = label_group.variables[actual_label_key][:]
            if label.ndim == 3:
                label = label[0]
            label = label.astype(np.uint8)
            label = tf.convert_to_tensor(label[..., np.newaxis], dtype=tf.uint8)
            label = tf.image.resize(label, target_size, method='nearest')

            input_list.append(inputs.numpy())
            label_list.append(label.numpy())

        except Exception as e:
            print(f"❌ Failed to load {path} —  {e}")
            continue

    if not input_list:
        raise RuntimeError("❌ No valid data found.")

    shapes = set(tuple(arr.shape) for arr in input_list)
    if len(shapes) != 1:
        raise ValueError("Shape mismatch in input arrays.")

    return (
        tf.convert_to_tensor(np.stack(input_list), dtype=tf.float32),
        tf.convert_to_tensor(np.stack(label_list), dtype=tf.uint8)
    )

# ✅ TF Dataset 생성
def create_tf_dataset(X, y, batch_size=4, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ✅ UNet 모델 정의
def build_unet_binary(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    x1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    p1 = tf.keras.layers.MaxPooling2D()(x1)

    x2 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(p1)
    p2 = tf.keras.layers.MaxPooling2D()(x2)

    b = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(p2)

    u1 = tf.keras.layers.UpSampling2D()(b)
    c1 = tf.keras.layers.Concatenate()([u1, x2])
    x3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(c1)

    u2 = tf.keras.layers.UpSampling2D()(x3)
    c2 = tf.keras.layers.Concatenate()([u2, x1])
    x4 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(c2)

    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(x4)

    return tf.keras.Model(inputs, outputs)

# ✅ 예측 및 시각화 함수
def visualize_all_with_prediction(X, y_true, y_pred, index, channel_names, iou_score=None):
    num_channels = X.shape[-1]
    rows, cols = 2, (num_channels + 2)

    plt.figure(figsize=(cols * 3, rows * 3))

    for c in range(num_channels):
        plt.subplot(rows, cols, c + 1)
        plt.title(channel_names[c])
        plt.imshow(X[index, :, :, c], cmap='viridis')
        plt.axis('off')
        plt.colorbar()

    plt.subplot(rows, cols, num_channels + 1)
    plt.title("Ground Truth")
    plt.imshow(y_true[index, :, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()

    plt.subplot(rows, cols, num_channels + 2)
    plt.title("Prediction")
    plt.imshow(y_pred[index, :, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()

    if iou_score is not None:
        plt.figtext(0.5, 0.05, f"IoU: {iou_score:.4f}", ha="center", fontsize=12, color="black")

    plt.tight_layout()

def evaluate_testset_with_visuals(X, y_true, y_pred, channel_names):
    print("🔍 Test 데이터 중 샘플 테스트\n")
    for i in range(len(X)):
        print(f"🖼  Sample {i}")
        gt = y_true[i].numpy().flatten()
        pred = y_pred[i].flatten()
        iou = jaccard_score(gt, pred)
        visualize_all_with_prediction(X, y_true, y_pred, index=i, channel_names=channel_names, iou_score=iou)
        print(f"IoU: {iou:.4f}\n{'-'*40}")

# ✅ 학습 데이터 경로 설정 (수정된 부분)
train_dir = "/data0/aix23606/soyeon/climatenet_full_train"
test_dir = "/data0/aix23606/soyeon/climatenet_full_test"  # 추가된 부분
train_files = sorted(glob(os.path.join(train_dir, '*.nc')))
test_files = sorted(glob(os.path.join(test_dir, '*.nc')))  # 추가된 부분

# ✅ 데이터 로딩 (기존 데이터 로딩 함수 재사용)
selected_vars = ['U850', 'V850', 'PSL', 'TMQ']
X_train, y_train = load_nc_data(train_files, selected_vars)
X_test, y_test = load_nc_data(test_files, selected_vars)  # 수정된 부분

# ✅ 라벨 이진화
y_train_bin = tf.cast(tf.where(y_train == 2, 1, 0), tf.uint8)
y_test_bin = tf.cast(tf.where(y_test == 2, 1, 0), tf.uint8)  # 추가된 부분

# ✅ 학습용 Dataset
train_dataset = create_tf_dataset(X_train, y_train_bin, batch_size=4)  # batch size 4로 수정
test_dataset = create_tf_dataset(X_test, y_test_bin, batch_size=4)  # 추가된 부분

# ✅ 모델 정의 및 학습
model = build_unet_binary(X_train.shape[1:])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ 학습 진행
history = model.fit(train_dataset, epochs=1000)

# ✅ 예측 수행
y_pred_prob = model.predict(X_test)
y_pred_bin = (y_pred_prob > 0.5).astype('float32')

# ✅ 시각화 - 랜덤으로 20개 샘플 선택 (수정된 부분)
random_indices = random.sample(range(len(X_test)), 20)  # 20개 샘플 랜덤 선택
X_test_subset = tf.gather(X_test, random_indices)
y_test_subset = tf.gather(y_test_bin, random_indices)
y_pred_subset = tf.gather(y_pred_bin, random_indices)

# ✅ IoU 계산
iou_scores = []
for i in range(len(X_test_subset)):
    gt = y_test_subset[i].numpy().flatten()
    pred = y_pred_subset[i].numpy().flatten()
    iou = jaccard_score(gt, pred)
    iou_scores.append(iou)

# ✅ IoU 평균 계산
mean_iou = np.mean(iou_scores)

# ✅ 시각화 + 저장 함수
def save_visualization(X, y_true, y_pred, index, channel_names, save_path, iou_score=None):
    """
    X: 입력 데이터
    y_true: 실제 레이블
    y_pred: 예측된 레이블
    index: 해당 샘플 인덱스
    channel_names: 채널 이름
    save_path: 이미지 저장 경로
    iou_score: IoU 점수 (옵션)
    """
    num_channels = X.shape[-1]
    rows, cols = 2, (num_channels + 2)

    plt.figure(figsize=(cols * 3, rows * 3))

    # 채널별 이미지 시각화
    for c in range(num_channels):
        plt.subplot(rows, cols, c + 1)
        plt.title(channel_names[c])
        plt.imshow(X[index, :, :, c], cmap='viridis')
        plt.axis('off')
        plt.colorbar()

    # 실제 값 시각화
    plt.subplot(rows, cols, num_channels + 1)
    plt.title("Ground Truth")
    plt.imshow(y_true[index, :, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()

    # 예측 값 시각화
    plt.subplot(rows, cols, num_channels + 2)
    plt.title("Prediction")
    plt.imshow(y_pred[index, :, :, 0], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.colorbar()

    # IoU 점수 추가
    if iou_score is not None:
        plt.figtext(0.5, 0.05, f"IoU: {iou_score:.4f}", ha="center", fontsize=12, color="black")

    # 이미지 저장
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ✅ 시각화 및 저장
output_dir = "/home/aix23606/soyeon/ar/result/train_test_v1"  # output_dir 경로 수정
os.makedirs(output_dir, exist_ok=True)

for i in range(len(X_test_subset)):
    save_path = os.path.join(output_dir, f"sample_{i}_viz.png")
    gt = y_test_subset[i].numpy().flatten()
    pred = y_pred_subset[i].numpy().flatten()
    iou = iou_scores[i]
    save_visualization(X_test_subset, y_test_subset, y_pred_subset, index=i,
                       channel_names=selected_vars, save_path=save_path, iou_score=iou)

# ✅ IoU 평균 저장
with open(os.path.join(output_dir, 'mean_iou.txt'), 'w') as f:
    f.write(f"Mean IoU: {mean_iou:.4f}")

print(f"✅ 시각화 저장 완료! → {output_dir}")