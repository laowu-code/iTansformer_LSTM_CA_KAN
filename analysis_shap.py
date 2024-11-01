import argparse
import time
import pandas as pd
import dill
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
import shap
import tqdm
from models import iTransformer_LSTM

warnings.filterwarnings('ignore')
from data import split_data_cnn,data_detime
from utils.tools import metrics_of_pv, EarlyStopping, same_seeds


def train(data, model, criterion, optm,  device=torch.device("cuda:0")):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for x, y in tqdm(data, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
        model.zero_grad()
        x, y = x.float().to(device), y.float().to(device)
        optm.zero_grad()
        y_pre = model(x)
        # loss = lpls_loss(y_pre, y)
        # loss=MLE_Gaussian(y_pre, y)
        loss = criterion(y_pre, y)
        loss.backward()
        optm.step()
        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(data.dataset)

    return epoch_loss

def evaluate(data, model, criterion,  device=torch.device("cuda:0"),scalar=None):
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    all_preds = []
    all_labels = []
    for x, y in tqdm(data):
        model.zero_grad()
        with torch.no_grad():
            x, y = x.float().to(device), y.float().to(device)
            # optm.zero_grad()
            y_pre = model(x)
            # loss = lpls_loss(y_pre, y)
            # loss=MLE_Gaussian(y_pre, y)
            loss = criterion(y_pre, y)
            # loss.backward()
            # optm.step()
            val_running_loss += loss.item() * x.size(0)
            all_preds.extend(y_pre.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    epoch_loss = val_running_loss / len(data.dataset)

    if scalar is not None:
        all_preds=scalar.inverse_transform(all_preds)
        all_labels = scalar.inverse_transform(all_labels)
    metrics_ = metrics_of_pv(all_preds, all_labels)
    return epoch_loss, metrics_

def slide_windows(nums, history):
    n, m = nums.shape
    ans = np.zeros([n - history + 1, history, m])
    for i in range(n - history + 1):
        ans[i] = nums[i:i + history]
    return ans


if __name__ == "__main__":
    seeds = 42
    same_seeds(seeds)
    # site='1B'
    site = '7-First-Solar'
    dataset = 'Spring'
    # dataset = 'Summer'
    # dataset = 'Autumn'
    # dataset = 'Winter'
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=150)
    # parser.add_argument('--data_dir', type=str, default='./dataset', help='数据集的路径')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    # file_path = f'./data/{site}/{dataset}_{site}_2017_2020_h.csv'
    file_path = f'./data/{site}/{dataset}_{site}_2019_2022_h.csv'
    num_nodes = 5
    epoch = 200
    batch = 300
    time_length = 24 * 1
    # predict_length = [1,4]
    predict_length = 1
    device = torch.device('cuda:0')
    df_all = pd.read_csv(file_path, header=0)
    multi_steps = False

    data_train, data_valid, data_test, timestamp_train, timestamp_valid, timestamp_test, scalar = split_data_cnn(df_all, 0.8, 0.1, time_length)
    dataset_train = data_detime(data=data_train, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)
    dataset_valid = data_detime(data=data_valid, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)
    dataset_test = data_detime(data=data_test, lookback_length=time_length, multi_steps=multi_steps, lookforward_length=predict_length)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    # model = Model_Torder(input_size=5).to(device)
    params_dict = {'hidden_dim': 32, 'layer_L': 3, 'layer_I': 4, 'heads': 12, 'dim_lstm': 32}
    model = iTransformer_LSTM(input_size=5, length_input=time_length, dim_embed=params_dict['hidden_dim'], dim_lstm=params_dict['dim_lstm'],
                           depth=params_dict['layer_I'], heads=params_dict['heads'], depth_lstm=params_dict['layer_L']).to(device)
    criterion_MAE = nn.L1Loss(reduction='sum').to(device)  # MAE
    criterion_MSE = nn.MSELoss(reduction='sum').to(device)  # MSE
    optm = optim.Adam(model.parameters(), lr=learning_rate)
    optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optm, mode="min", factor=0.5, patience=5, verbose=True
    )
    model_name = f"iTransformer_TCN_parallel_MAE_{dataset}"
    model_save = f"model_save/{dataset}/{model_name}.pt"
    train_losses, valid_losses = [], []
    earlystopping = EarlyStopping(model_save, patience=10, delta=0.0001)

    # if not os.path.exists(f'data_record/{dataset}/{model_name}.csv'):
    #     os.makedirs(f'data_record/{dataset}/{model_name}.csv'')
    # need_train = True
    need_train = False
    model_save = f"model_save/{dataset}/{model_name}.pt" if need_train else f"model_save/{dataset}/best/{model_name}.pt"
    if need_train:
        try:
            for epoch in range(epochs):
                time_start = time.time()
                train_loss = train(data=train_loader, model=model, criterion=criterion_MAE,
                                   optm=optm, batch_size=batch_size,
                                   )
                valid_loss, ms = evaluate(
                    data=valid_loader, model=model, criterion=criterion_MAE, batch_size=batch_size
                )
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                optm_schedule.step(valid_loss)
                earlystopping(valid_loss, model)
                # torch.save(model, model_save, pickle_module=dill)
                print('')
                print(f'{model_name}|time:{(time.time() - time_start):.2f}|Loss_train:{train_loss:.4f}|Learning_rate:{optm.state_dict()["param_groups"][0]["lr"]:.4f}\n'
                      f'Loss_valid:{valid_loss:.4f}|MAE:{ms[0]:.4f}|RMSE:{ms[1]:.4f}|R2:{ms[2]:.4f}|MBE:{ms[3]:.4f}', flush=True, )
                if earlystopping.early_stop:
                    print("Early stopping")
                    break  # 跳出迭代，结束训练
        except KeyboardInterrupt:
            print("Training interrupted by user")
        plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
        plt.plot(np.arange(len(valid_losses)), valid_losses, label="valid rmse")
        plt.legend()  # 显示图例
        plt.xlabel("epoches")
        # plt.ylabel("epoch")
        plt.title("Train_loss&Valid_loss")
        plt.show()
with open(model_save, "rb") as f:
    model = torch.load(f, pickle_module=dill)
    # print(model)
# 选择100个样本作为背景数据
x_train = slide_windows(data_train, time_length)
device = torch.device('cpu')
model = model.to(device)
# model.eval()
# 选择背景样本并调整形状
background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
background = background.reshape(100, -1)

# 创建 SHAP 解释器
def model_predict(data):
    # 将输入转换为 Tensor 并调整为 3D 形状 (样本数, 24, 5)
    data_tensor = torch.tensor(data, dtype=torch.float32).reshape(-1, 24, 5).to(device)
    with torch.no_grad():
        # 进行预测，并将结果转换为 NumPy 数组返回
        return model(data_tensor).cpu().numpy()

explainer = shap.KernelExplainer(model_predict, background)

# 解释前 10 个样本
X_sample = x_train[:10].reshape(10, -1)  # 获取前 10 个样本，形状为 (10, 24, 5)
# shap_values = explainer.shap_values(X_sample)
# np.save('data_record/shap.npy',np.array(shap_values))
shap_values=np.load('data_record/shap.npy')
print(shap_values.shape)
# 对特征重要性进行汇总，展示每个特征（在所有时间步上的累积影响）
# 注意 shap_values 的第一个维度应与 X_sample 的第一个维度相同
shap_values = shap_values.reshape(10, 24, 5)
# shape_values=
shap.plots.waterfall(shap_values[0])


# model = model.to(device)
# test_loss, ms_test = evaluate(
#     data=test_loader, model=model, criterion=criterion_MAE, batch_size=batch_size,scalar=scalar
# )
# print(f'Test_valid:{test_loss:.4f}|MAE:{ms_test[0]:.4f}|RMSE:{ms_test[1]:.4f}|R2:{ms_test[2]:.4f}|MBE:{ms_test[3]:.4f}')
# with open(f'data_record/{dataset}/Metrics_{model_name}.csv', 'a', encoding='utf-8', newline='') as f:
#     csv_write = csv.writer(f)
#     csv_write.writerow([f'{site}_pred1_{model_name}', ms_test[0], ms_test[1], ms_test[2], ms_test[3]])

# device = torch.device('cpu')
# device=torch.device('cpu')
# model = model.to(device)
# model.eval()
#
# # 定义函数，通过钩子获取特征图和梯度
# def forward_with_hook(model, x,idx=0):
#     feature_maps = []
#     grads = []
#
#     # 定义钩子函数，用于获取特征图
#     def forward_hook_fn(module, input, output):
#         feature_maps.append(output)
#
#     # 定义钩子函数，用于获取梯度
#     def backward_hook_fn(module, grad_input, grad_output):
#         grads.append(grad_output[0])
#
#     # 注册前向和后向钩子到 cross 层
#     handle_forward = model.cross.register_forward_hook(forward_hook_fn)
#     handle_backward = model.cross.register_full_backward_hook(backward_hook_fn)
#
#     # 获取模型输出
#     output = model(x)
#
#     # 选择目标输出的第一个元素，进行反向传播
#     output_idx = idx
#     output[output_idx].backward(retain_graph=True)
#
#     # 移除钩子
#     handle_forward.remove()
#     handle_backward.remove()
#
#     return feature_maps[0], grads[0]
#
# # 获取输入数据并转换为 tensor
# x = torch.tensor(x_train[0:10], dtype=torch.float32).to(device)
# x.requires_grad = True  # 设置为 True 以便计算梯度
#
# idx=4
# # 获取特征图和梯度
# feature_map, grads = forward_with_hook(model, x,idx=idx)
#
# # 对梯度在特征维度上进行平均，以获得特征图的权重
# weights = torch.mean(grads, dim=(1)).cpu().numpy()  # 假设时间维度在 axis=2
#
# # 计算加权特征图并生成 CAM
# # 获取第一个样本的特征图和权重
# feature_map = feature_map[idx].detach().cpu().numpy()  # 变为 (1, 64) -> (64,)
# weights = weights[idx].reshape(-1,1)  # (64,)
#
# # 对特征图进行加权求和，生成 CAM
# cam = np.dot(weights, feature_map)  # 对 feature_map 加权求和，结果为 (time_steps,)
#
# # 使用 ReLU 将负值过滤掉，保留正相关部分
# cam = np.maximum(cam, 0)
#
# # 将 CAM 归一化到 [0, 1] 范围
# cam = cam - np.min(cam)
# cam = cam / np.max(cam)
# cam_tensor = torch.tensor(cam, dtype=torch.float32)  # 将 cam 转换为 tensor
# cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度以适配 pooling 函数
#
# # 选择合适的池化层，将 64x64 缩减到 64x24（聚合到时间步）
# pool = torch.nn.AdaptiveAvgPool2d((64, 24))
# cam_pooled = pool(cam_tensor).squeeze().numpy()  # 结果形状为 (64, 24)
#
# # 进一步聚合特征维度，将 64x24 聚合到 5x24，以适应输入数据的形状
# # 这里使用均值，也可以使用 max pooling 或其他方式
# cam = cam_pooled[0:5]
# # 可视化 CAM
# cam_normalized = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # 归一化到 [0, 1]
#
# # 创建图形
# plt.figure(figsize=(10, 6))
#
# # 可视化原始数据
# plt.imshow(x_train[idx].reshape(5,-1), aspect='auto', cmap='viridis', alpha=0.3)  # 原始数据使用 viridis 颜色映射
# plt.colorbar(label='Original Data Value')  # 添加颜色条
#
# # 可视化 CAM
# plt.imshow(cam_normalized, aspect='auto', cmap='plasma', alpha=0.5)  # CAM 使用 plasma 颜色映射
#
# plt.title('Overlay of Original Data and Class Activation Map (CAM)')
# plt.xlabel('Time Step')
# plt.ylabel('Features')
#
# # 调整布局
# plt.tight_layout()
# plt.show()

