import argparse
import csv
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import TCN_iTransformer
from data import split_data_cnn,data_detime
from utils.tools import metrics_of_pv, EarlyStopping, same_seeds
import optuna
import warnings
warnings.filterwarnings('ignore')


def train(data, model, criterion, optm, batch_size=64, device=torch.device("cuda:0")):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for x, y in tqdm(data, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
        model.zero_grad()
        x, y = x.float().to(device), y.float().to(device)
        optm.zero_grad()
        y_pre = model(x)
        # loss = lpls_loss(y_pre, y)
        # loss = MLE_Gaussian(y_pre, y)
        loss = criterion(y_pre, y)
        loss.backward()
        optm.step()
        running_loss += loss.item() * x.size(0)

    epoch_loss = running_loss / len(data.dataset)

    return epoch_loss


def evaluate(data, model, criterion, batch_size=64, device=torch.device("cuda:0")):
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
            # loss = MLE_Gaussian(y_pre, y)
            loss = criterion(y_pre, y)
            # loss.backward()
            # optm.step()
            val_running_loss += loss.item() * x.size(0)
            all_preds.extend(y_pre.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    epoch_loss = val_running_loss / len(data.dataset)

    metrics_ = metrics_of_pv(all_preds, all_labels)
    return epoch_loss, metrics_




if __name__ == "__main__":

    seeds = 42
    same_seeds(seeds)
    # site='1B'
    site = '7-First-Solar'
    dataset = 'Spring'
    parser = argparse.ArgumentParser(description="Hyperparameters")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
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
    time_length = 24 * 2
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


    def objective(trial):
        # model = Model_Torder(input_size=5).to(device)
        channels = trial.suggest_categorical('channels', [16, 32, 64])
        kernel_size = trial.suggest_categorical('kernel_size', [2, 3, 4])
        dim_embed = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128, 256, 512])
        layer_T = trial.suggest_categorical('layer_T', [1, 2, 3, 4])
        layer_I = trial.suggest_categorical('layer_I', [1, 2, 3, 4, 5, 6])
        heads = trial.suggest_categorical('heads', [2, 4, 6, 8, 12, 16])
        dim_mlp = trial.suggest_int('dim_mlp', 10, 30)
        # num_channels = [32] * 1, kernel_size = 3, length_pre = 1,
        # length_input = 48, dim_embed = 128, depth = 4, heads = 6, dim_mlp = 10,
        model = TCN_iTransformer(input_size=5, num_channels=[channels] * layer_T, kernel_size=kernel_size,
                             dim_embed=dim_embed, depth=layer_I, heads=heads, dim_mlp=dim_mlp).to(device)
        criterion_MAE = nn.L1Loss(reduction='sum').to(device)  # MAE
        criterion_MSE = nn.MSELoss(reduction='sum').to(device)  # MSE
        optm = optim.Adam(model.parameters(), lr=learning_rate)
        optm_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optm, mode="min", factor=0.5, patience=5, verbose=True
        )

        model_name = "iTransformer_TCN_MAE"
        model_save = f"model_save/{dataset}/{model_name}.pt"
        train_losses, valid_losses = [], []
        earlystopping = EarlyStopping(model_save, patience=10, delta=0.0001)

        # if not os.path.exists(f'data_record/{dataset}/{model_name}.csv'):
        #     os.makedirs(f'data_record/{dataset}/{model_name}.csv'')
        need_train = True
        # need_train = False

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
            # plt.plot(np.arange(len(train_losses)), train_losses, label="train loss")
            # plt.plot(np.arange(len(valid_losses)), valid_losses, label="valid rmse")
            # plt.legend()  # 显示图例
            # plt.xlabel("epoches")
            # # plt.ylabel("epoch")
            # plt.title("Train_loss&Valid_loss")
            # plt.show()
        test_loss, ms_test = evaluate(
            data=test_loader, model=model, criterion=criterion_MAE, batch_size=batch_size
        )
        print(f'Test_valid:{test_loss:.4f}|MAE:{ms_test[0]:.4f}|RMSE:{ms_test[1]:.4f}|R2:{ms_test[2]:.4f}|MBE:{ms_test[3]:.4f}', )
        with open(f'data_record/{dataset}/Metrics_{model_name}_{dataset}.csv', 'a', encoding='utf-8', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow([f'{site}_pred1_{model_name}', ms_test[0], ms_test[1], ms_test[2], ms_test[3]])
        return ms_test[0]
        # with open(model_save, "rb") as f:
        #     model = torch.load(f, pickle_module=dill)
        # model = model.to(device)
        # test_loss,ms_test = evaluate(
        #     data=test_loader, model=model,criterion=criterion_MAE, batch_size=batch_size
        # )
        # print(f'Test_valid:{test_loss:.4f}|MAE:{ms_test[0]:.4f}|RMSE:{ms_test[1]:.4f}|R2:{ms_test[2]:.4f}|MBE:{ms_test[3]:.4f}',)
        # with open(f'data_record/{dataset}/Metrics_{model_name}_{dataset}.csv', 'a', encoding='utf-8',newline='') as f:
        #     csv_write = csv.writer(f)
        #     csv_write.writerow([f'{site}_pred1_{model_name}', ms_test[0],ms_test[1],ms_test[2],ms_test[3]])


    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(), load_if_exists=True,
                                storage=f'sqlite:///MAE_.sqlite3', study_name=f'{site}_{dataset}_{predict_length}_single_MAE_10_10')
    study.optimize(objective, n_trials=100)
    print(study.best_params, '\n', study.best_value)

# optuna-dashboard sqlite:///MAE_.sqlite3
