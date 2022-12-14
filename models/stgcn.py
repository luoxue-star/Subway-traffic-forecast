import torch
from torch import nn

import sys

sys.path.append("../")


class TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size))
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        """
        :param X: (batch_size, num_nodes, num_steps, num_features)
        :return: (batch_size, num_node, num_steps, num_features)
        """
        # (batch_size, num_nodes, num_steps, num_features) -> (batch_size, num_features, num_nodes, num_steps)
        X = X.permute(0, 3, 1, 2)
        Y = self.conv1(X) + self.sigmoid(self.conv2(X))
        output = self.relu(Y + self.conv3(X))
        return output.permute(0, 2, 3, 1)  # (batch_size, num_node, num_steps, num_features)


class TransformerTimeBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(TransformerTimeBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.encoder1 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.encoder2 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.encoder3 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        """
        :param X: (batch_size, num_steps, num_nodes, features)
        :return: (batch_size, num_nodes, num_steps, hidden_size)
        """
        bs, num_steps, num_nodes = X.shape[:3]
        X = X.permute(0, 2, 1, 3)
        X = self.fc1(X)
        X = X.reshape(bs, num_steps*num_nodes, -1)
        Y = self.encoder1(X) + self.sigmoid(self.encoder2(X))
        output = self.relu(Y + self.encoder3(X))
        return output.reshape(bs, num_steps, num_nodes, -1)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_channels, num_nodes):
        super(STGCNBlock, self).__init__()
        self.time_block1 = TimeBlock(in_channels, out_channels)
        self.W = nn.Linear(out_channels, spatial_channels)
        self.time_block2 = TimeBlock(in_channels=spatial_channels, out_channels=out_channels)
        self.bn = nn.BatchNorm2d(num_nodes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X, A):
        """
        :param X: (batch_size, num_nodes, num_steps, num_features)
        :param A: (num_nodes, num_nodes) 邻接矩阵
        :return: (batch_size, num_nodes, num_steps, num_features)
        """
        t1 = self.time_block1(X)  # (batch_size, num_node, num_steps, num_features)
        # lfs:(num_nodes, batch_size, num_steps, num_features)
        lfs = torch.einsum("ij, jklm->kilm", [A, t1.permute(1, 0, 2, 3)])
        t2 = self.relu(self.W(lfs))
        t3 = self.time_block2(t2)
        return self.bn(t3)


class STTransformerBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, spatial_size, num_nodes):
        super(STTransformerBlock, self).__init__()
        self.time_block1 = TransformerTimeBlock(input_size=input_size, hidden_size=hidden_size, num_heads=num_heads)
        self.W = nn.Linear(hidden_size, spatial_size)
        self.time_block2 = TransformerTimeBlock(input_size=spatial_size, hidden_size=hidden_size, num_heads=num_heads)
        self.bn = nn.BatchNorm2d(num_nodes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """
        :param X: (batch_size, num_steps, num_nodes, num_features)
        :param A: (num_nodes, num_nodes) 邻接矩阵
        :return: (batch_size, num_steps, num_nodes, hidden_size)
        """
        t1 = self.time_block1(X)  # (batch_size, num_steps, num_nodes, hidden_size)
        # bs, num_steps, num_nodes, hidden_size = t1.shape
        # (batch_size, num_steps, num_nodes, hidden_size)
        lfs = torch.einsum("ij, jklm -> kilm", [A, t1.permute(2, 0, 1, 3)])
        t2 = self.relu(self.W(lfs))  # (batch_size, num_nodes, num_steps, spatial_size)
        t3 = self.time_block2(t2)  # (batch_size, num_nodes, num_steps, hidden_size)
        return self.bn(t3).permute(0, 2, 1, 3)  # (batch_size, num_steps, num_nodes, hidden_size)


class STGCN(nn.Module):
    def __init__(self, num_steps, num_nodes, num_features, hidden_size, output_days):
        super(STGCN, self).__init__()
        self.block1 = STGCNBlock(in_channels=num_features, out_channels=hidden_size,
                                 spatial_channels=hidden_size // 4, num_nodes=num_nodes)
        self.block2 = STGCNBlock(in_channels=hidden_size, out_channels=hidden_size,
                                 spatial_channels=hidden_size // 4, num_nodes=num_nodes)
        self.time_block = TimeBlock(in_channels=hidden_size, out_channels=hidden_size)
        self.fc = nn.Linear((num_steps - 2 * 5) * hidden_size, output_days)

    def forward(self, X, A):
        X = X.permute(0, 2, 1, 3)
        Y = self.block1(X, A)  # (batch_size, num_nodes, num_steps, num_features)
        Y = self.block2(Y, A)  # (batch_size, num_nodes, num_steps, num_features)
        Y = self.time_block(Y)  # (batch_size, num_nodes, num_steps, num_features)
        Y = self.fc(Y.reshape((Y.shape[0], Y.shape[1], -1)))
        return Y


class STTransformer(nn.Module):
    def __init__(self, num_steps, num_nodes, num_heads, num_features, hidden_size, output_size=2):
        super(STTransformer, self).__init__()
        self.block1 = STTransformerBlock(input_size=num_features, hidden_size=hidden_size, num_heads=num_heads,
                                         spatial_size=hidden_size // 4, num_nodes=num_nodes)
        self.block2 = STTransformerBlock(input_size=hidden_size, hidden_size=hidden_size, num_heads=num_heads,
                                         spatial_size=hidden_size // 4, num_nodes=num_nodes)
        self.time_block = TransformerTimeBlock(input_size=hidden_size, hidden_size=hidden_size, num_heads=num_heads)
        self.fc = nn.Linear(num_steps * hidden_size, output_size)  # ????

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """
        :param X: (batch_size, num_steps, num_nodes, num_features)
        :param A: (num_nodes, num_nodes)
        :return:
        """
        Y = self.block1(X, A)  # (batch_size, num_steps, num_nodes, num_features)
        Y = self.block2(Y, A)  # (batch_size, num_steps, num_nodes, num_features)
        Y = self.time_block(Y)  # (batch_size, num_steps, num_nodes, num_features)
        Y = self.fc(Y.reshape(Y.shape[0], Y.shape[2], -1))  # (batch_size, num_nodes, num_features*num_steps)
        return Y  # (batch_size, num_nodes, 2)


# if __name__ == "__main__":
#     X = torch.rand(16, 14, 147, 2)  # (batch_size, num_steps, num_nodes, num_features)
#     net = STTransformer(num_steps=14, num_nodes=147, num_features=2,
#                         hidden_size=256, num_heads=4)
#     A = torch.rand(147, 147)
#     print(net(X, A).shape)
