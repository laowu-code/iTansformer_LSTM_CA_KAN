import pandas as pd
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import dill
from tqdm import tqdm
def metrics_of_pv(preds, trues):
    pred = np.array(preds)
    true = np.array(trues)
    mae = np.round(mean_absolute_error(true, pred),4)
    rmse = np.round(np.sqrt(mean_squared_error(true, pred)),4)
    r2 = np.round(r2_score(true, pred),4)
    mbe = np.round(np.mean(pred - true),4)
    # sMAPE = np.round(100 * np.mean(np.abs(preds - trues) / (np.abs(preds) + np.abs(trues))), 4)
    return [mae,rmse,r2,mbe]
def save2csv(n, file):
    n = n.reshape((1, n.shape[0]))
    n = pd.DataFrame(n)
    n.to_csv(file, index=False, encoding='utf-8', header=False, mode='a')
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = True  # GPU、网络结构固定，可设置为True
    # torch.backends.cudnn.deterministic = True  # 固定网络结构
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        # path = os.path.join(self.save_path, 'best_network.pth')
        path = self.save_path
        torch.save(model, path, pickle_module=dill)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

def train(data, model, criterion, optm, device=torch.device("cuda:0")):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(data):
        model.zero_grad()
        x, y = x.float().to(device), y.float().to(device)
        optm.zero_grad()
        y_pre = model(x)
        loss = criterion(y_pre, y)
        loss.backward()
        optm.step()
        running_loss += loss.item() * x.size(0)
    epoch_loss = running_loss / len(data.dataset)
    return epoch_loss
def evaluate(data, model, criterion, device=torch.device("cuda:0"),scalar=None):
    model.eval()
    val_running_loss = 0.0
    all_preds = []
    all_labels = []
    for x, y in tqdm(data):
        model.zero_grad()
        with torch.no_grad():
            x, y = x.float().to(device), y.float().to(device)
            y_pre = model(x)
            loss = criterion(y_pre, y)
            val_running_loss += loss.item() * x.size(0)
            all_preds.extend(y_pre.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    epoch_loss = val_running_loss / len(data.dataset)
    if scalar is not None:
        all_preds=scalar.inverse_transform(all_preds)
        all_labels = scalar.inverse_transform(all_labels)
    metrics_ = metrics_of_pv(all_preds, all_labels)
    return epoch_loss, metrics_