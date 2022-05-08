import datetime
import math
import torch.nn.functional as F
from utils import *
from torch import nn
from torch.nn import Module


class ItemConv(Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.b = nn.Parameter(torch.Tensor(self.layers + 1, 1, 1))
        torch.nn.init.ones_(self.b)

    def forward(self, adjacency, embedding):
        adjacency = adj_to_spare(adjacency)
        item_embeddings = embedding
        final = [item_embeddings.unsqueeze(0)]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings.unsqueeze(0))
        final = F.normalize(torch.cat(final, dim=0), dim=-1, p=2)
        item_embeddings = torch.sum(self.b * final, dim=0)
        return item_embeddings


class GatConv(Module):
    def __init__(self, batch_size, alpha, emb_size=100):
        super(GatConv, self).__init__()
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.a_0 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, item_embeddings, adj):
        h = item_embeddings
        N = h.shape[1]
        a_input = (h.repeat(1, 1, N).view(self.batch_size, N * N, self.emb_size)
                   * h.repeat(1, N, 1)).view(self.batch_size, N, N, self.emb_size)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(self.batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(self.batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(self.batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(self.batch_size, N, N)

        adj = trans_to_cuda(adj)
        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)

        alpha = torch.softmax(alpha, dim=-1)
        output = torch.matmul(alpha, h)

        return output


class SessionConv(Module):
    def __init__(self, layers, emb_size=100):
        super(SessionConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.a = nn.Parameter(torch.Tensor(self.layers + 1, 1, 1))
        torch.nn.init.ones_(self.a)

    def forward(self, adjacency, embedding):
        adjacency = adj_to_spare(adjacency)
        item_embeddings = embedding
        final = [item_embeddings.unsqueeze(0)]
        for i in range(self.layers):
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(item_embeddings.unsqueeze(0))
        final = F.normalize(torch.cat(final, dim=0), dim=-1, p=2)
        item_embeddings = torch.sum(self.a * final, dim=0)
        return item_embeddings


class GnnCl(Module):
    def __init__(self, gnn_adjacency, emb_size, n_node, num_layers, batch_size, lr):
        super().__init__()
        self.gnn_adjacency = gnn_adjacency
        self.lr = lr
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.n_node = n_node
        self.nodes_emb = nn.Embedding(self.n_node, self.emb_size)

        self.Gnn = ItemConv(self.num_layers)
        self.Gat = GatConv(self.batch_size, alpha=0.2)
        self.Gnn_s = SessionConv(self.num_layers)

        self.g_w1 = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))
        self.normal = nn.BatchNorm1d(self.batch_size)
        self.relu = nn.ReLU(inplace=True)
        self.g_w2 = nn.Parameter(torch.Tensor(self.emb_size, self.emb_size))

        self.pos_embedding = nn.Embedding(150, self.emb_size)
        self.w_1 = nn.Linear(2 * self.emb_size, self.emb_size)
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_seq(self, item_embedding, reversed_sess_item, h, alias_inputs, alpha=0.):
        reversed_sess_item = trans_to_cuda(torch.LongTensor(reversed_sess_item))
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, reversed_sess_item.shape[1], self.emb_size).fill_(0)
        for i in torch.arange(reversed_sess_item.shape[0]):
            seq_h[i] = get(i)

        get = lambda index: h[index][alias_inputs[index]]
        seq_h_ = torch.stack([get(i) for i in torch.arange(alias_inputs.shape[0]).long()])

        seq_h = (1 - alpha) * seq_h + alpha * F.dropout(seq_h_, 0.5, self.training)

        return seq_h

    def soft_attention(self, seq_h, mask, session_len):

        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = self.w_1(torch.cat([pos_emb, seq_h], -1))
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)

        return select

    def g(self, features):
        features = torch.matmul(self.g_w1, features)
        features = self.normal(features)
        self.relu(features)
        features = torch.matmul(self.g_w2, features)
        return features

    def contrastive_learning(self, features):
        labels = torch.cat([torch.arange(features.shape[0] / 2) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = trans_to_cuda(labels)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        flag = trans_to_cuda(torch.eye(labels.shape[0], dtype=torch.bool))
        labels = labels[~flag].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~flag].view(similarity_matrix.shape[0], -1)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = trans_to_cuda(torch.zeros(logits.shape[0], dtype=torch.long))
        logits = logits / 0.2
        con_loss = self.loss_function(logits, labels)
        return 0.01 * con_loss

    def forward(self, tar, reversed_sess_item, mask, session_len, session_adj, is_train):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)

        item_embeddings_gnn = self.Gnn(self.gnn_adjacency, self.nodes_emb.weight)
        item_embeddings_gnn_1 = F.dropout(item_embeddings_gnn, 0.3, training=self.training)
        item_embeddings_gnn_1 = torch.cat([zeros, item_embeddings_gnn_1], 0)

        item_embeddings_gnn_2 = F.dropout(item_embeddings_gnn, 0.1, training=self.training)
        item_embeddings_gnn_2 = torch.cat([zeros, item_embeddings_gnn_2], 0)

        embeddings = item_embeddings_gnn_1 + item_embeddings_gnn_2

        alias_session, adj, items = data_masks_3(reversed_sess_item)

        h_1 = item_embeddings_gnn_1[items]
        h_1 = self.Gat(h_1, adj)
        h_1 = self.get_seq(item_embeddings_gnn_1, reversed_sess_item, h_1, alias_session, alpha=0.8)

        h_2 = item_embeddings_gnn_2[items]
        h_2 = self.Gat(h_2, adj)
        h_2 = self.get_seq(item_embeddings_gnn_2, reversed_sess_item, h_2, alias_session, alpha=0.8)

        select_gnn_1 = self.soft_attention(h_1, mask, session_len)
        select_gnn_1 = self.Gnn_s(session_adj, select_gnn_1)

        select_gnn_2 = self.soft_attention(h_2, mask, session_len)
        select_gnn_2 = self.Gnn_s(session_adj, select_gnn_2)

        if is_train:
            select_hg_features = self.g(select_gnn_1)
            select_gnn_features = self.g(select_gnn_2)

            features = torch.cat([select_hg_features, select_gnn_features], dim=0)
            con_loss = self.contrastive_learning(features)
        else:
            con_loss = 0

        seq_h = select_gnn_1 + select_gnn_2

        scores = torch.mm(seq_h, torch.transpose(embeddings[1:], 1, 0))
        loss = self.loss_function(scores + 1e-8, tar)

        return con_loss, loss, scores


def processing(model, i, data, is_train):
    tar, session_len, session_item, reversed_sess_item, mask, session_adj = data.get_slice(i)
    tar = trans_to_cuda(torch.LongTensor(tar))
    session_len = trans_to_cuda(torch.LongTensor(session_len))
    mask = trans_to_cuda(torch.LongTensor(mask))

    if is_train:
        con_loss, tar_loss, scores = model(tar, reversed_sess_item, mask, session_len, session_adj, is_train=True)
        return con_loss, tar_loss
    else:
        con_loss, tar_loss, scores = model(tar, reversed_sess_item, mask, session_len, session_adj, is_train=False)
        return tar, scores


def train_test(model, train_data, test_data, batch_size):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    model.train()
    slices = train_data.generate_batch(batch_size)
    for b, i in enumerate(slices):
        model.zero_grad()
        con_loss, tar_loss, = processing(model, i, train_data, is_train=True)
        loss = con_loss + tar_loss
        print(b, con_loss.item(), tar_loss.item())
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, hit_, mrr_ = [], [], [], []
    slices = test_data.generate_batch(batch_size)
    for i in slices:
        tar, scores = processing(model, i, test_data, is_train=False)
        tar = trans_to_cpu(tar).detach().numpy()
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        for score, target in zip(sub_scores, tar):
            hit.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target)[0][0] + 1))

        sub_scores_ = scores.topk(10)[1]
        sub_scores_ = trans_to_cpu(sub_scores_).detach().numpy()
        for score, target in zip(sub_scores_, tar):
            hit_.append(np.isin(target, score))
            if len(np.where(score == target)[0]) == 0:
                mrr_.append(0)
            else:
                mrr_.append(1 / (np.where(score == target)[0][0] + 1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100

    hit_ = np.mean(hit_) * 100
    mrr_ = np.mean(mrr_) * 100

    return hit, mrr, hit_, mrr_
