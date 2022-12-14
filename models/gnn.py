import copy
import random

import torch
from torch import nn
import torch.nn.functional as F

from utils.data_preprocess import *

DEVICE = torch.device("cuda:3")


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, alpha: float, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_dim
        self.out_features = out_dim
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(self.alpha)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param features: (num_nodes, num_feature)
        :param adj_matrix: (num_node, num_node)
        :return:
        """
        Wh = torch.mm(features, self.W)  # Wh:(num_nodes, out_dim)
        # 这一步是核心，获取到注意力机制的输入
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (num_node, num_node, 2 * out_dim)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))  # (num_node, num_node)
        zero_matrix = -1e6 * torch.ones_like(e)
        # softmax会把很小的数直接置0
        attention = torch.where(adj_matrix > 0, e, zero_matrix)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # 注意力系数加权求和
        values = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(values)  # elu激活函数
        else:
            return values

    def _prepare_attentional_mechanism_input(self, Wh: torch.Tensor) -> torch.Tensor:
        num_node = Wh.size()[0]  # num_node
        # 将每一个节点特征进行重复，为了后面进行拼接
        Wh_repeat1 = Wh.repeat_interleave(num_node, dim=0)  # (num_node*num_node, out_dim)
        Wh_repeat2 = Wh.repeat(num_node, 1)  # (num_node*num_node, out_dim)
        # 两两拼接
        concat_matrix = torch.cat([Wh_repeat1, Wh_repeat2], dim=1)  # (N * N, 2 * out_features)
        return concat_matrix.view(num_node, num_node, 2 * self.out_features)


class GAT(nn.Module):
    def __init__(self, dim_feature, hidden_size, num_output, dropout, alpha, num_heads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        # 多个图注意力层
        self.attentions = [GATLayer(dim_feature, hidden_size, dropout=dropout, alpha=alpha, concat=True)
                           for _ in range(num_heads)]
        # 将多个层添加起来
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        # 输出层
        self.output = GATLayer(hidden_size * num_heads, num_output, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        # 将多个注意力层的输出拼接起来
        features = torch.cat([att(features, adj_matrix) for att in self.attentions], dim=1)
        features = F.dropout(features, self.dropout, training=self.training)
        features = self.output(features, adj_matrix)
        return features


class GCN(nn.Module):
    def __init__(self, in_dim: int, out_dim=7):
        super(GCN, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim * 2, bias=False)
        self.fc2 = nn.Linear(in_dim * 2, in_dim // 2, bias=False)
        self.fc3 = nn.Linear(in_dim // 2, out_dim, bias=False)

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        :param features: (num_nodes, batch_size, num_steps, features)
        :param adj_matrix: (num_nodes, num_nodes)
        :return:
        """
        features = F.relu(self.fc1(torch.einsum("ij, jklm -> iklm", adj_matrix, features)))
        features = F.relu(self.fc2(torch.einsum("ij, jklm -> iklm", adj_matrix, features)))
        return self.fc3(torch.einsum("ij, jklm -> iklm", adj_matrix, features))


# class GCN(nn.Module):
#     def __init__(self, in_dim: int, out_dim=7):
#         super(GCN, self).__init__()
#         self.fc1 = nn.Linear(in_dim, in_dim * 2, bias=False)
#         self.fc2 = nn.Linear(in_dim * 2, in_dim // 2, bias=False)
#         self.fc3 = nn.Linear(in_dim // 2, out_dim, bias=False)
#
#     def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
#         features = F.relu(self.fc1(torch.einsum("ij, jkl -> ikl", adj_matrix, features)))
#         features = F.relu(self.fc2(torch.einsum("ij, jkl -> ikl", adj_matrix, features)))
#         return self.fc3(torch.einsum("ij, jkl -> ikl", adj_matrix, features))


# def train_and_eval(features, adj_matrix, labels, epochs=25):
#     net.to(DEVICE)
#     features = features.to(DEVICE)
#     labels = labels.to(DEVICE)
#     adj_matrix = adj_matrix.to(DEVICE)
#     for epoch in range(epochs):
#         net.train()
#         optimizer.zero_grad()
#         pred = net(features, adj_matrix)[train_index]
#         loss = loss_func(pred, labels[train_index])
#         loss.backward()
#         optimizer.step()
#
#         print(f"[epoch:{epoch + 1}/{epochs}], train loss:{loss.item()}")
#
#         net.eval()
#         with torch.no_grad():
#             pred = net(features, adj_matrix)[val_index]
#             loss = loss_func(pred, labels[val_index])
#
#         print(f"val loss:{loss.item()}")


# if __name__ == "__main__":
#     hsr_adj, hsr_inflow, hsr_outflow = load_data("../data/HSR_adj.csv",
#                                                  "../data/HSR_inflow.csv",
#                                                  "../data/HSR_outflow.csv")
#     # features = torch.eye(hsr_adj.shape[0])  # 使用one-hot编码作为初始的特征矩阵
#     # features = torch.load("../save_embeddings/save.embeddings.pt")
#     features = torch.zeros_like(hsr_adj)
#     index = torch.where(hsr_adj != 0)  # 找出边的位置
#     values = (hsr_adj[index] - torch.mean(hsr_adj[index])) / torch.std(hsr_adj[index])
#     features[index] = values   # 获得标准化后的边信息
#
#     hsr_adj = torch.where(hsr_adj != 0, 1, 0)  # 邻接矩阵
#
#     labels = copy.deepcopy(hsr_adj)  # 用邻接矩阵直接作为输出的标签
#
#     net = GAT(dim_feature=features.shape[1], hidden_size=512, num_output=len(labels),
#               dropout=0.1, alpha=0.05, num_heads=4)
#     train_len = int(0.8 * len(labels))
#     index = list(range(0, len(labels)))
#     random.shuffle(index)
#     train_index = index[:train_len]
#     val_index = index[train_len:]
#
#     optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
#     loss_func = nn.SmoothL1Loss()
#
#     train_and_eval(features, hsr_adj, labels, epochs=300)
