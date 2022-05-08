import torch
import numpy as np
import random
from scipy.sparse import coo_matrix
from sklearn.manifold import TSNE
from visdom import Visdom


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# adj for gnn
def data_masks_2(all_sessions, n_node):
    adj = dict()
    for sess in all_sessions:
        for i in range(len(sess) - 1):
            if sess[i] - 1 not in adj.keys():
                adj[sess[i] - 1] = dict()
                adj[sess[i] - 1][sess[i] - 1] = 1
                adj[sess[i] - 1][sess[i + 1] - 1] = 1
            else:
                if sess[i + 1] - 1 not in adj[sess[i] - 1].keys():
                    adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    adj[sess[i] - 1][sess[i] - 1] += 1
                else:
                    adj[sess[i] - 1][sess[i + 1] - 1] += 1
                    adj[sess[i] - 1][sess[i] - 1] += 1
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))
    return coo


def data_masks_3(sessions):
    adj = []
    alias_session = []
    items = []

    for session in sessions:
        adj_1 = np.zeros((len(session), len(session)))
        node = np.unique(session)
        item = node.tolist() + (len(session) - len(node)) * [0]

        for i in range(len(session) - 1):
            u = np.where(node == session[i])[0][0]
            adj_1[u][u] = 1
            if session[i + 1] == 0:
                break

            v = np.where(node == session[i + 1])[0][0]
            if u == v or adj_1[u][v] == 4:
                continue
            adj_1[v][v] = 1
            if adj_1[v][u] == 2:
                adj_1[u][v] = 4
                adj_1[v][u] = 4
            else:
                adj_1[u][v] = 2
                adj_1[v][u] = 3
        adj.append(torch.tensor(adj_1))

        alias_session_1 = [np.where(node == i)[0][0] for i in session]
        alias_session.append(torch.tensor(alias_session_1))

        items.append(torch.tensor(item))

    adj = torch.stack(adj)
    alias_session = torch.stack(alias_session)
    items = torch.stack(items)
    return alias_session, adj, items


class Data:
    def __init__(self, data, all_train_seq, shuffle=False, n_node=None):
        self.raw = np.asarray(data[0])
        self.shuffle = shuffle
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.gnn_edge = data_masks_2(all_train_seq, n_node)

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)

        session_len = []
        reversed_sess_item = []
        mask = []

        session_adj = dict()
        for i, session in enumerate(inp):
            session_adj[i] = dict()
            session_adj[i][i] = 1
            for j, _session in enumerate(inp):
                if j >= i:
                    break
                else:
                    session_adj[i][j] = len(set(session).intersection(_session)) / len(session)
                    session_adj[j][i] = len(set(session).intersection(_session)) / len(_session)

                    if self.targets[index][i] == self.targets[index][j]:
                        session_adj[i][j] += 1
                        session_adj[j][i] += 1

            nonzero_elems = np.nonzero(session)[0]
            session_len.append([len(nonzero_elems)])
            # pad 0 at behind
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            # [1, 1, ... ,1 , 0 , 0, ...]
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])

        row, col, data = [], [], []
        for i in session_adj.keys():
            item = session_adj[i]
            for j in item.keys():
                row.append(i)
                col.append(j)
                data.append(session_adj[i][j])
        session_adj = coo_matrix((data, (row, col)), shape=(len(session_adj), len(session_adj)))
        return self.targets[index] - 1, session_len, items, reversed_sess_item, mask, session_adj


def adj_to_spare(adjacency):
    values = adjacency.data
    indices = np.vstack((adjacency.row, adjacency.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adjacency.shape
    adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return adjacency


def show_embeddings(embeddings, win):
    index = random.sample(range(0, embeddings.shape[0]), int(embeddings.shape[0] / 10))
    embeddings = embeddings[index]
    tsne = TSNE(n_components=2, init='pca')
    tsne_embeddings = tsne.fit_transform(embeddings)
    vis = Visdom()
    vis.scatter(tsne_embeddings, win=win, env="main", opts=dict(markersize=1))
