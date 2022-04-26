#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import networkx as nx
from tensorflow import keras
from tensorflow.keras import layers
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.data
from kgcnn.data.cora.cora import cora_graph
from kgcnn.literature.GCN import make_gcn
from kgcnn.utils.adj import precompute_adjacency_scaled, convert_scaled_adjacency_to_list, make_adjacency_undirected_logical_or
from kgcnn.utils.data import ragged_tensor_from_nested_numpy
from kgcnn.utils.learning import lr_lin_reduction


# In[ ]:


from dgl.nn import GraphConv

class GCN(nn Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)


# In[ ]:


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    for e in range(100):
        logits = model(g, features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
train(g, model)


# In[ ]:


A_data, X_data, y_data = cora_graph()
nodes = X_data.todense()
plt.figure(figsize=(12,8))
plt.plot(np.arange(1, len(trainlossall) + 1), trainlossall, label='Training Loss', c='blue')
plt.plot(np.arange(epostep, epo + epostep, epostep), testlossall[:, 1], label='Test Loss', c='red')
plt.xlabel('Epochs')
plt.ylabel('Accurarcy')
plt.title('GCN')
plt.legend(loc='lower right', fontsize='x-large')
plt.savefig('gcn_loss.png')
plt.show()

