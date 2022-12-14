import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


def load_data(adj_path="./data/HSR_adj.csv",
              inflow_path="./data/HSR_inflow.csv",
              outflow_path="./data/HSR_outflow.csv"):
    """
    加载数据并进行缺失值处理
    :return: 邻接矩阵和处理好的数据
    """
    hsr_adj = pd.read_csv(adj_path, header=None)
    hsr_adj = torch.from_numpy(np.array(hsr_adj, dtype=np.float32))
    hsr_inflow = pd.read_csv(inflow_path, header=None)
    hsr_inflow = torch.from_numpy(np.asarray(fill_nan_by_mean(hsr_inflow), dtype=np.float32))
    hsr_outflow = pd.read_csv(outflow_path, header=None)
    hsr_outflow = torch.from_numpy(np.asarray(fill_nan_by_mean(hsr_outflow), dtype=np.float32))
    return hsr_adj, hsr_inflow, hsr_outflow


def fill_nan_by_mean(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    对含有确实值的列，用每一列的平均值进行缺失值填充
    :param dataframe: 数据
    :return:
    """
    columns_na = dataframe.columns[dataframe.isnull().sum(axis=0) > 0]  # 找出含缺失值的列
    for column in list(columns_na):
        mean = int(dataframe[column].mean())  # 因为次数只能是均值
        dataframe[column].fillna(mean, inplace=True)
    return dataframe


def standard_scale(data: torch.Tensor):
    """
    数据标准化
    :param data: 输入数据
    :return: 用于标准化和逆标准化的类、标准化好的数据
    """
    data = data.numpy()
    ss = StandardScaler()
    data = ss.fit_transform(data)
    return ss, torch.from_numpy(data)


def create_dataset(features: torch.Tensor, look_back: int, train_ratio=0.875):
    """
    进行数据集的分割
    :param features: 输入数据
    :param look_back: 记录过去几天的信息
    :param train_ratio: 训练集的比例
    :return: 训练数据和验证数据
    """
    X, y = [], []
    for i in range(len(features) - look_back - 1):
        X.append(features[i: i + look_back, :].numpy())
        y.append(features[i + look_back, :].numpy())

    train_len = int(len(X) * train_ratio)
    X, y = torch.from_numpy(np.asarray(X)), torch.from_numpy(np.asarray(y))
    # # (num_samples, num_steps, num_features) -> (num_samples, num_features, num_steps)
    # X = X.reshape(X.shape[0], X.shape[2], X.shape[1])
    return X[:train_len, :, :], y[:train_len, :], X[train_len:, :, :], y[train_len:, :]


def inverse_scale(data: np.ndarray, scale_in, scale_out, type: str):
    """
    将标准化数据转化回去
    :param data: 输入数据包含进站和出站
    :param scale_in: 用于进站数据逆变换的实例变量
    :param scale_out: 用于出站数据逆变换的实例变量
    :param type:
    :return: 逆转后的数据
    """
    if type == "flatten":
        data_in = data[:, :int(data.shape[1] / 2)]
        data_out = data[:, int(data.shape[1] / 2):]
    elif type == "stack":
        data_in = data[:, 0::2]
        data_out = data[:, 1::2]
    data_in = scale_in.inverse_transform(data_in)
    data_out = scale_out.inverse_transform(data_out)
    return np.concatenate((data_in, data_out), axis=1)


class InOutFlowDatasetFlatten(Dataset):
    """
    用于常规的时序模型和时刻之间的注意力机制模型
    """

    def __init__(self, inflow_features, outflow_features, inflow_labels, outflow_labels):
        super(InOutFlowDatasetFlatten, self).__init__()
        self.features = torch.cat((inflow_features, outflow_features), dim=2)  # 将进站和出站的特征拼接起来
        self.labels = torch.cat((inflow_labels, outflow_labels), dim=1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


class InOutFlowDatasetStack(Dataset):
    """
    用于STGCN和站内计算注意力机制的时序模型
    """
    def __init__(self, inflow_features, outflow_features, inflow_labels, outflow_labels):
        super(InOutFlowDatasetStack, self).__init__()
        self.features = torch.stack((inflow_features, outflow_features), dim=3)
        self.labels = torch.stack((inflow_labels, outflow_labels), dim=2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.features[item], self.labels[item]


# if __name__ == "__main__":
#     adj_matrix, inflow, outflow = load_data(adj_path="../data/HSR_adj.csv",
#                                             inflow_path="../data/HSR_inflow.csv",
#                                             outflow_path="../data/HSR_outflow.csv")
#     ss_in, std_inflow = standard_scale(inflow)
#     ss_out, std_outflow = standard_scale(outflow)
#
#     inflow_train_features, inflow_train_labels, inflow_val_features, inflow_val_labels = create_dataset(std_inflow,
#                                                                                                         7)
#     outflow_train_features, outflow_train_labels, outflow_val_features, outflow_val_labels = create_dataset(std_outflow,
#                                                                                                             7)
#     train_dataset = InOutFlowDatasetFlatten(inflow_train_features, outflow_train_features,
#                                             inflow_train_labels, outflow_train_labels)
#     train_iter = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False)
#     val_dataset = InOutFlowDatasetFlatten(inflow_val_features, outflow_val_features, inflow_val_labels,
#                                           outflow_val_labels)
#     val_iter = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False)
#
#     for X, y in train_iter:
#         print(X.shape)
#         print(y.shape)
