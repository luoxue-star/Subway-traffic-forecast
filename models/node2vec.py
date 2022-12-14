import torch
from torch_geometric.nn import Node2Vec
from scipy import sparse

import pandas as pd
import numpy as np
from tqdm import tqdm

DEVICE = torch.device("cuda:2")
torch.manual_seed(1024)


def train_node2vec(epochs=10):
    model.to(DEVICE)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in tqdm(data_loader):
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(DEVICE), neg_rw.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch:{epoch+1}, loss:{total_loss / len(data_loader)}")


if __name__ == "__main__":
    adj_matrix = pd.read_csv("../data/HSR_adj.csv", header=None)

    sparse_matrix = sparse.coo_matrix(adj_matrix)  # 转化为稀疏矩阵的存储形式
    edge_index = torch.from_numpy(np.array([sparse_matrix.row,
                                            sparse_matrix.col], dtype=int))  # 得到边列表
    model = Node2Vec(edge_index, embedding_dim=64, walk_length=20, context_size=5, p=0.2,
                     q=5, sparse=True)
    data_loader = model.loader(batch_size=32, shuffle=False)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.0001)
    train_node2vec()

    embeddings = model(torch.arange(len(adj_matrix.columns), device=DEVICE))
    print(embeddings.shape)
    print(embeddings)

    torch.save(embeddings.detach().cpu(), "../save_embeddings/save.embeddings.pt")







