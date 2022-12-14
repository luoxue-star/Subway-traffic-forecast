import torch
from torch import nn

import math


class LSTM(nn.Module):
    def __init__(self, num_input, num_hidden, num_output,
                 use_attention=False, attention_type="station_with_station", attention_score_type="DotProduct"):
        """
        :param num_input: 输入特征的维度
        :param num_hidden: 输入的隐藏层维度
        :param num_output: 输出的维度
        :param use_attention: 是否使用attention计算注意力分数后再输入到LSTM中
        :param attention_type: 计算注意力的方式
        :param attention_score_type: 计算注意力分数的方式
        """
        super(LSTM, self).__init__()
        self.use_attention = use_attention
        assert attention_type == "station_with_station" or attention_type == "time_with_time"
        self.attention_type = attention_type
        if self.use_attention:
            if attention_type == "station_with_station":  # 每个站点之间计算注意力分数
                self.attention = MultiHeadAttention(hidden_size=2, num_heads=1,
                                                    query_size=2, key_size=2, value_size=2,
                                                    attention_score_type=attention_score_type)
                self.lstm = nn.LSTM(input_size=2*147, hidden_size=num_hidden, bias=True,
                                    batch_first=True)
            else:  # 每个时间点之间计算注意力分数
                self.attention = MultiHeadAttention(hidden_size=num_hidden, num_heads=1,
                                                    query_size=num_input, key_size=num_input, value_size=num_input,
                                                    attention_score_type=attention_score_type)
                self.lstm = nn.LSTM(input_size=num_hidden, hidden_size=num_hidden, bias=True,
                                    batch_first=True)
        else:  # 直接使用LSTM模型进行预测
            self.lstm = nn.LSTM(input_size=num_input, hidden_size=num_hidden, bias=True,
                                batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(num_hidden, num_output)

    def forward(self, X):
        if self.use_attention:
            if self.attention_type == "station_with_station":
                attn_out = []
                for i in range(X.shape[1]):
                    attn_out.append(self.attention(X[:, i, :, :], X[:, i, :, :], X[:, i, :, :]))
                attn_out = torch.stack(attn_out, dim=1).flatten(2)
                y, (h, c) = self.lstm(attn_out)
                return self.fc(h.squeeze(0)).reshape(h.shape[1], -1, 2)
            else:
                attn_out = self.attention(X, X, X)
                y, (h, c) = self.lstm(attn_out)

        else:
            y, (h, c) = self.lstm(X)
        return self.fc(h.squeeze(0))


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, query_size, key_size, value_size, attention_score_type="Additive"):
        super(MultiHeadAttention, self).__init__()
        """使用并行计算计算多头注意力，这里必须保证hidden_size/num_heads是一个整数"""
        self.num_heads = num_heads
        assert attention_score_type == "DotProduct" or attention_score_type == "Additive"
        if attention_score_type == "DotProduct":
            self.attention = DotProductAttention()
        elif attention_score_type == "Additive":
            self.attention = AdditiveAttention(hidden_size=int(hidden_size / num_heads))
        self.W_q = nn.Linear(query_size, hidden_size, bias=False)
        self.W_k = nn.Linear(key_size, hidden_size, bias=False)
        self.W_v = nn.Linear(value_size, hidden_size, bias=False)
        self.W_o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, queries, keys, values):
        # q,k,v:(batch_size, num_steps, hidden_size)->(batch_size*num_heads, num_steps, hidden_size/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # output:(batch_size*num_heads,查询个数，hidden_size/num_heads)
        output = self.attention(queries, keys, values)
        # output_concat:(batch_size, 查询个数， hidden_size) # 相当于把多个头拼接起来了
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, queries, keys, values):
        """
        :param queries:(batch_size, 查询个数, 维度)
        :param keys: (batch_size, 键值对个数, 维度)
        :param values: (batch_size, 键值对个数, 维度)
        :return: (batch_size, 查询个数， 维度)  最后输出是由查询的个数决定的
        """
        # scores:(batch_size, 查询个数, 键值对个数)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(keys.shape[2])
        scores = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(scores, values)


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(AdditiveAttention, self).__init__()
        self.w = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, queries, keys, values):
        # queries、keys、values：(batch_size, 个数, hidden_size)
        Y = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))  # (batch_size, 个数, 个数, hidden_size)
        # scores:(batch_size, 个数, 个数)
        # 利用了广播机制
        scores = self.w(Y).squeeze(-1)
        scores = nn.functional.softmax(scores, dim=-1)
        return torch.bmm(scores, values)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，hidden_size)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # hidden_sizes/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

    # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
    # hidden_sizes/num_heads)
    X = X.permute(0, 2, 1, 3)

    # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
    # hidden_size/num_heads)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    # 输入X:(batch_size*num_heads,查询个数，hidden_size/num_heads)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])  # X:(batch_size,num_heads,查询个数，hidden_size/num_heads)
    # X:(batch_size,查询个数，num_heads, hidden_size/num_heads)
    X = X.permute(0, 2, 1, 3)
    # return:(batch_size, 查询个数， hidden_size) # 相当于把多个头拼接起来了
    return X.reshape(X.shape[0], X.shape[1], -1)


class Transformer(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, num_head):
        super(Transformer, self).__init__()
        self.trans = nn.Linear(num_input, num_hidden)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.encoder = nn.TransformerEncoderLayer(d_model=num_hidden, nhead=num_head, batch_first=True)
        self.decoder = nn.TransformerDecoderLayer(d_model=num_hidden, nhead=num_head, batch_first=True)
        self.fc = nn.Linear(num_hidden, num_output)

    def forward(self, X):
        X = self.leaky_relu(self.trans(X))
        target = X[:, -1:, :]  # 作为decoder的输入
        input = X[:, :-1, :]  # 作为encoder的输入
        memory = self.encoder(input)
        output = self.decoder(target, memory).squeeze(1)
        return self.fc(output)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_head, output_size):
        super(EncoderLSTM, self).__init__()
        self.trans = nn.Linear(input_size, hidden_size)
        self.ac = nn.ELU(inplace=True)
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_head, batch_first=True)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, bias=True, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X):
        X = self.ac(self.trans(X))
        Y = self.encoder(X)
        Y, (H, C) = self.lstm(Y)
        return self.fc(H.squeeze(0))


# if __name__ == "__main__":
#     X = torch.rand(32, 7, 294)
#     net = Transformer(294, 512, 294, 8)
#     net(X)
