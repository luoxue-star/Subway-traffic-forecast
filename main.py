from typing import List

import numpy as np

from models.sequence_models import *
from models.stgcn import *
from utils.data_preprocess import *
from models.sequence_model_with_gnn import *

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

DEVICE = torch.device("cuda:5")
torch.manual_seed(42)


def train(epochs=30):
    net.to(DEVICE)
    for epoch in range(epochs):
        net.train()
        l = 0
        num = 0
        for X, y in tqdm(train_iter):
            optimizer.zero_grad()
            if net.__class__.__name__ == "STGCN" or net.__class__.__name__ == "EncoderLSTMWithGCN" or \
                    net.__class__.__name__ == "EncoderLSTMWithGAT" or net.__class__.__name__ == "LSTMWithGCN" or \
                    net.__class__.__name__ == "STTransformer":
                pred = net(X.to(DEVICE), adj_matrix.to(DEVICE))
            else:
                pred = net(X.to(DEVICE))
            loss = loss_func(pred, y.to(DEVICE))
            loss.backward()
            optimizer.step()
            l += loss.item() * len(y)
            num += len(y)

        print(f"epoch:{epoch + 1}, loss:{l / num}")
    torch.save(net.state_dict(), "./save_weights/EncoderLSTMwithGAT_epoch_{}.pth".format(epochs + 180))
    val()


def val():
    preds = []
    labels = []
    net.eval()
    with torch.no_grad():
        l = 0
        num = 0
        for X, y in tqdm(val_iter):
            if net.__class__.__name__ == "STGCN" or net.__class__.__name__ == "EncoderLSTMWithGCN" or \
                    net.__class__.__name__ == "EncoderLSTMWithGAT" or net.__class__.__name__ == "LSTMWithGCN" or \
                    net.__class__.__name__ == "STTransformer":
                pred = net(X.to(DEVICE), adj_matrix.to(DEVICE))
            else:
                pred = net(X.to(DEVICE))
            loss = loss_func(pred, y.to(DEVICE))
            l += loss.item() * len(y)
            num += len(y)
            preds.append(pred.detach().cpu().numpy().flatten())
            labels.append(y.numpy().flatten())

        print(f"loss:{l / num}")
    # plot_results(preds, labels, index=0, type="stack")
    plot_results(preds, labels, index=0, type="flatten")


def main():
    train(epochs=5)


def plot_results(preds: List[np.ndarray], labels: List[np.ndarray], index: int, type="flatten"):
    """
    绘制图像
    :param preds: 这里的数据包含进站和出站的数据的预测结果
    :param labels: 真实值
    :param index : 绘制图像的地铁站索引
    :param type: 表示数据的形式
    :return: None
    """
    preds = np.asarray(preds, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    preds = inverse_scale(preds, ss_in, ss_out, type)
    labels = inverse_scale(labels, ss_in, ss_out, type)
    plot(preds[:, index], labels[:, index])
    # np.save("./save_results/pred_by_LSTM.npy", preds)
    # np.save("./save_results/labels_in_LSTM.npy", labels)

    # np.save("./save_results/pred_by_STGCN.npy", preds)
    # np.save("./save_results/labels_in_STGCN.npy", labels)

    # np.save("./save_results/pred_by_EncoderLSTM.npy", preds)
    # np.save("./save_results/labels_in_EncoderLSTM.npy", labels)

    # np.save("./save_results/pred_by_Transformer.npy", preds)
    # np.save("./save_results/labels_in_Transformer.npy", labels)

    # np.save("./save_results/pred_by_EncoderLSTMWithGCN.npy", preds)
    # np.save("./save_results/labels_in_EncoderLSTMWithGCN.npy", labels)

    np.save("./save_results/pred_by_EncoderLSTMWithGAT.npy", preds)
    np.save("./save_results/labels_in_EncoderLSTMWithGAT.npy", labels)

    # np.save("./save_results/pred_by_STTransformer.npy", preds)
    # np.save("./save_results/labels_in_STTransformer.npy", labels)


def plot(preds, labels):
    plt.figure()
    plt.plot(range(1, len(preds) + 1), preds, label='predict')
    plt.plot(range(1, len(labels) + 1), labels, label="label")
    plt.legend()
    plt.xlabel("day")
    plt.ylabel("flow")
    plt.show()


def eval(preds: np.ndarray, labels: np.ndarray):
    mse = metrics.mean_squared_error(labels, preds)
    mae = metrics.mean_absolute_error(labels, preds)
    mape = np.mean(np.abs((preds - labels) / labels))
    # print(f"MSE:{mse}, MAE:{mae}, mape:{mape}")
    return mse, mae, mape


if __name__ == "__main__":
    adj_matrix, inflow, outflow = load_data()  # 加载数据
    adj_matrix = torch.where(adj_matrix != 0, 1.0, 0.0)
    adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0])
    degree = torch.sum(adj_matrix, dim=1)
    degree_matrix = torch.diag(degree ** (-0.5))
    adj_matrix = torch.matmul(degree_matrix, adj_matrix).mm(degree_matrix)
    ss_in, std_inflow = standard_scale(inflow)  # 数据标准化
    ss_out, std_outflow = standard_scale(outflow)
    #
    # 创建数据集并封装为dataloader
    inflow_train_features, inflow_train_labels, inflow_val_features, inflow_val_labels = create_dataset(std_inflow,
                                                                                                        8)
    outflow_train_features, outflow_train_labels, outflow_val_features, outflow_val_labels = create_dataset(std_outflow,
                                                                                                            8)
    train_dataset = InOutFlowDatasetFlatten(inflow_train_features, outflow_train_features,
                                            inflow_train_labels, outflow_train_labels)
    val_dataset = InOutFlowDatasetFlatten(inflow_val_features, outflow_val_features, inflow_val_labels, outflow_val_labels)

    # 创建数据集并封装为dataloader
    # train_dataset = InOutFlowDatasetStack(inflow_train_features, outflow_train_features, inflow_train_labels,
    #                                       outflow_train_labels)
    # val_dataset = InOutFlowDatasetStack(inflow_val_features, outflow_val_features, inflow_val_labels,
    #                                     outflow_val_labels)

    train_iter = DataLoader(train_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=8)
    val_iter = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=8)

    # net = LSTM(num_input=294, num_hidden=512, num_output=294, use_attention=False)  # flatten 有用
    # net = Transformer(num_input=294, num_hidden=512, num_output=294, num_head=8)  # flatten
    # net = STGCN(num_steps=12, num_nodes=147, num_features=2,
    #             hidden_size=128, output_days=2)  # stack
    # net = LSTM(num_input=294, num_hidden=512, num_output=294, use_attention=True, attention_type="time_with_time")
    # net = LSTM(num_input=294, num_hidden=512, num_output=294,
    #            use_attention=True, attention_type="station_with_station")  # LSTM with attention(station with station)
    # net = EncoderLSTM(input_size=294, hidden_size=512, num_head=4, output_size=294)  # 有用，batch_size=32，lr=0.0001
    # net = EncoderLSTMWithGCN(input_size=294, hidden_size=32, output_size=294, num_heads=4)  # flatten， 12天，lr=0.001，8batch
    net = EncoderLSTMWithGAT(294, 64, 294, 8, 8)  # flatten
    # net = STTransformer(num_steps=10, num_nodes=147, num_features=2,
    #                     hidden_size=128, num_heads=4)
    # net = LSTMWithGCN(vec_size=2, step_size=10, hidden_size=16)  # stack
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    loss_func = nn.SmoothL1Loss()

    net_dict = torch.load("./save_weights/EncoderLSTMwithGAT_epoch_180.pth", map_location="cpu")
    net.load_state_dict(net_dict)
    main()

    pred = np.load("./save_results/pred_by_EncoderLSTMWithGAT.npy")
    label = np.load("./save_results/labels_in_EncoderLSTMWithGAT.npy")
    mse, mae, mape = 0, 0, 0
    for i in range(pred.shape[1]):
        # print(f"station:{i}", end=' ')
        mse_, mae_, mape_ = eval(pred[:, i], label[:, i])
        mse += mse_
        mae += mae_
        mape += mape_
        # plot(pred[:, i], label[:, i])
    print(f"average mse:{mse / pred.shape[1]}, "
          f"average mae:{mae / pred.shape[1]}, "
          f"average mape:{mape / pred.shape[1]}")

