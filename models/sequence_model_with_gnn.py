import torch

from models.gnn import *
from models.sequence_models import *


class EncoderLSTMWithGCN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_heads: int):
        super(EncoderLSTMWithGCN, self).__init__()
        self.trans = nn.Linear(input_size, hidden_size)
        self.ac = nn.ELU(inplace=True)
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        # hidden_size=input_size是为了符合GCN输入
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True)
        self.gcn = GCN(2, hidden_size)
        # 147表示节点个数
        self.lstm2 = nn.LSTM(input_size=hidden_size * 147, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X: torch.Tensor, adj_matrix=torch.Tensor):
        """
        :param X: (batch_size, num_steps, num_nodes * 2)
        :param adj_matrix: （num_nodes, num_nodes)
        :return:
        """
        X = self.ac(self.trans(X))  # (batch_size, num_steps, hidden_size)
        Y = self.encoder(X)  # (batch_size, num_steps, hidden_size)
        Y, (H, C) = self.lstm1(Y)  # (batch_size, num_steps, input_size)
        bs, num_steps, num_nodes = Y.shape[0], Y.shape[1], Y.shape[2] // 2
        Y = Y.reshape(num_nodes, bs, num_steps, -1)
        Y = self.gcn(Y, adj_matrix)  # (num_nodes, bs, num_steps, hidden_size)
        Y = Y.reshape(bs, num_steps, -1)
        Y, (H, C) = self.lstm2(Y)
        return self.fc(H.squeeze(0))


class EncoderLSTMWithGAT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, attention_heads, gat_heads):
        super(EncoderLSTMWithGAT, self).__init__()
        self.trans = nn.Linear(input_size, hidden_size)
        self.ac = nn.ELU(inplace=True)
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=attention_heads, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=input_size, batch_first=True)
        self.gat = GAT(2, hidden_size=hidden_size, num_output=hidden_size, dropout=0.1, alpha=0.1, num_heads=gat_heads)
        self.lstm2 = nn.LSTM(input_size=hidden_size * 147, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.short_cut = nn.Linear(2, hidden_size)

    def forward(self, X: torch.Tensor, adj_matrix: torch.Tensor):
        Y = self.ac(self.trans(X))  # (batch_size, num_steps, hidden_size)
        Y = self.encoder(Y)  # (batch_size, num_steps, hidden_size)
        Y, (H, C) = self.lstm1(Y)  # (batch_size, num_steps, input_size)
        bs, num_steps, num_nodes = Y.shape[0], Y.shape[1], Y.shape[2] // 2
        Y = Y.reshape(num_nodes, bs * num_steps, -1)  # (num_node, bs * num_steps, 2)

        output = []
        for i in range(bs * num_steps):
            output.append(self.gat(Y[:, i, :], adj_matrix))
        output = torch.stack(output, dim=1)  # (num_nodes, bs * num_steps, hidden_size)

        output = output.reshape(bs, num_steps, -1)
        short_cut = self.ac(self.short_cut(X.reshape(bs, num_steps, num_nodes, -1))).reshape(bs, num_steps, -1)
        output = output + short_cut
        output, (H, C) = self.lstm2(output)
        return self.fc(H.squeeze(0))


# if __name__ == "__main__":
#     X = torch.rand(32, 12, 147, 2)
#     adj_matrix = torch.eye(147)
#     net = EncoderLSTMWithGCN(input_size=2, hidden_size=16, output_size=2)
#     print(net(X, adj_matrix).shape)




