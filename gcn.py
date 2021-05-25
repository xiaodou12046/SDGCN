import torch
import numpy as np
import copy
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.autograd import Variable
from tree import Tree, head_to_tree, tree_to_adj
from transformers import BertModel, BertConfig, BertPreTrainedModel, BertTokenizer


class GCNClassifier(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super(GCNClassifier, self).__init__()
        self.args = args
        self.in_dim = args.hidden_dim
        self.gcn_model = GCNAbsaModel(args, emb_matrix=emb_matrix)
        self.classifier = nn.Linear(self.in_dim, args.num_class)

    def forward(self, inputs):
        outputs, h_syn, h_sem = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, outputs, h_syn, h_sem


class GCNAbsaModel(nn.Module):
    def __init__(self, args, emb_matrix=None):
        super(GCNAbsaModel, self).__init__()
        self.args = args
        self.num_layers = args.num_layers

        # Bert
        if args.emb_type == "bert":
            config = BertConfig.from_pretrained(args.bert_model_dir)
            config.output_hidden_states = True
            self.bert = BertModel.from_pretrained(args.bert_model_dir, config=config, from_tf=False)

        emb_matrix = torch.Tensor(emb_matrix)
        self.emb_matrix = emb_matrix

        self.in_drop = nn.Dropout(args.input_dropout)

        # create embedding layers
        self.emb = nn.Embedding(args.token_vocab_size, args.emb_dim, padding_idx=0)
        if emb_matrix is not None:
            self.emb.weight = nn.Parameter(emb_matrix.to(self.args.device), requires_grad=False)

        self.pos_emb = nn.Embedding(args.pos_vocab_size, args.pos_dim, padding_idx=0) \
                                    if args.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(args.post_vocab_size, args.post_dim, padding_idx=0) \
                                    if args.post_dim > 0 else None  # position emb

        # rnn layer
        self.in_dim = args.emb_dim + args.post_dim + args.pos_dim
        if self.args.emb_type == 'bert':
            self.dense = nn.Linear(self.in_dim, args.rnn_hidden * 2)
        else:
            self.rnn = nn.LSTM(self.in_dim, args.rnn_hidden, 1, batch_first=True, bidirectional=True)

        # # multi-head attention
        self.cos_attn_adj = KHeadAttnCosSimilarity(args.head_num,
                                                    2 * args.hidden_dim,
                                                    args.threshold)

        # gcn layer
        self.gcn_syn = nn.ModuleList()
        self.gcn_sem = nn.ModuleList()
        # gate weight
        self.w_syn = nn.ParameterList()
        self.w_sem = nn.ParameterList()

        self.gcn_syn.append(GCN(args, args.rnn_hidden * 2, args.hidden_dim))
        self.gcn_sem.append(GCN(args, args.rnn_hidden * 2, args.hidden_dim))
        for i in range(1, self.args.num_layers):
            self.gcn_syn.append(GCN(args, args.hidden_dim, args.hidden_dim))
            self.gcn_sem.append(GCN(args, args.hidden_dim, args.hidden_dim))
            self.w_syn.append(nn.Parameter(
                torch.FloatTensor(args.hidden_dim, args.hidden_dim).normal_(0, 1)))
            self.w_sem.append(nn.Parameter(
                torch.FloatTensor(args.hidden_dim, args.hidden_dim).normal_(0, 1)))

        self.syn_attn = TimeWiseAspectBasedAttn(args.hidden_dim, args.num_layers)
        self.sem_attn = TimeWiseAspectBasedAttn(args.hidden_dim, args.num_layers)

        # AM attention
        self.attn = Attention(2 * args.hidden_dim, args.hidden_dim)

        # learnable hyperparameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # fully connect Layer
        self.linear = nn.Linear(2 * args.hidden_dim, args.hidden_dim)

    def create_embs(self, tok, pos, post):
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)
        return embs

    def create_bert_embs(self, tok, pos, post, word_idx, segment_ids):
        bert_outputs = self.bert(tok, token_type_ids=segment_ids)
        feature_output = bert_outputs[0]
        word_embs = torch.stack([torch.index_select(f, 0, w_i)
                               for f, w_i in zip(feature_output, word_idx)])
        embs = [word_embs]
        if self.args.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.args.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=-1)
        embs = self.in_drop(embs)
        return embs

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(self.args, batch_size, 1, True)
        rnn_inputs = pack_padded_sequence(rnn_inputs, seq_lens.cpu(), batch_first=True)
        rnn_outputs, (_, _) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def create_adj_mask(self, rnn_hidden):
        score_mask = torch.matmul(rnn_hidden, rnn_hidden.transpose(-2, -1))
        score_mask = (score_mask == 0)
        return score_mask

    def graph_comm(self, h0, w, h1, score_mask):
        # H = softmax(h1 * w * h2)
        H = torch.matmul(h0, w)
        H = torch.matmul(H, h1.transpose(-2, -1))
        H = H.masked_fill(score_mask, -1e10)  # masked
        b = ~score_mask[:, :, 0:1]
        H = F.softmax(H, dim=-1) * b.float()

        # h = h0 + H * h1
        h = h0 + torch.matmul(H, h1)
        return h

    def Dense(self, inputs, seq_lens):
        # padd
        inputs = self.dense(inputs)
        inputs_unpad = pack_padded_sequence(inputs, seq_lens.cpu(), batch_first=True)
        outputs, _ = pad_packed_sequence(inputs_unpad, batch_first=True)
        return outputs

    def forward(self, inputs):
        if self.args.emb_type == "bert":
            tok, asp, pos, head, post, dep, asp_mask, length, adj, word_idx, segment_ids = inputs
            embs = self.create_bert_embs(tok, pos, post, word_idx, segment_ids)
            hidden = self.Dense(embs, length)
        else:
            tok, asp, pos, head, post, dep, asp_mask, length, adj = inputs       # unpack inputs
            # embedding
            embs = self.create_embs(tok, pos, post)
            # bi-lstm encoding
            hidden = self.encode_with_rnn(embs, length, embs.size(0))  # [batch_size, seq_len, hidden]

        score_mask = self.create_adj_mask(hidden)

        # cosine adj matrix
        cos_adj = self.cos_attn_adj(hidden, score_mask)  # [batch_size, head_num, seq_len, seq_len]
        cos_adj = torch.sum(cos_adj, dim=1) / self.args.head_num

        h_syn = []
        h_sem = []
        # GCN encoding
        h_syn.append(self.gcn_syn[0](adj, hidden, score_mask, first_layer=True))
        h_sem.append(self.gcn_sem[0](cos_adj, hidden, score_mask, first_layer=True))
        for i in range(self.args.num_layers - 1):
            # graph communication layer
            h_syn_ = self.graph_comm(h_syn[i], self.w_syn[i], h_sem[i], score_mask)
            h_sem_ = self.graph_comm(h_sem[i], self.w_sem[i], h_syn[i], score_mask)

            h_syn.append(self.gcn_syn[i + 1](adj, h_syn_, score_mask, first_layer=False))
            h_sem.append(self.gcn_sem[i + 1](cos_adj, h_sem_, score_mask, first_layer=False))

        h_syn = torch.stack(h_syn, dim=0)
        h_sem = torch.stack(h_sem, dim=0)

        # time-wise aspect-based attention
        h_syn_final = self.syn_attn(h_syn[:-1], h_syn[-1], asp_mask, score_mask)
        h_sem_final = self.sem_attn(h_sem[:-1], h_sem[-1], asp_mask, score_mask)

        # h = torch.cat((h_syn_final, h_sem_final), dim=-1)
        h = self.alpha * h_syn_final + (1 - self.alpha) * h_sem_final

        # linear
        outputs = F.relu(self.linear(h))
        return outputs, h_syn_final, h_sem_final


def rnn_zero_state(args, batch_size, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, args.rnn_hidden)
    h0 = c0 = torch.zeros(*state_shape)
    return h0.to(args.device), c0.to(args.device)


class GCN(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(GCN, self).__init__()
        self.args = args
        self.drop = nn.Dropout(args.gcn_dropout)

        # gcn layer
        self.W = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, adj, inputs, score_mask, first_layer=True):
        # gcn
        denom = adj.sum(2).unsqueeze(2) + 1  # norm adj
        Ax = adj.bmm(inputs)
        AxW = self.W(Ax)
        AxW = AxW / denom
        gAxW = F.relu(AxW)
        out = gAxW if first_layer else self.drop(gAxW)
        return out


class KHeadAttnCosSimilarity(nn.Module):
    def __init__(self, head_num, input_dim, threshold):
        super(KHeadAttnCosSimilarity, self).__init__()
        assert (input_dim / head_num) != 0
        self.d_k = int(input_dim // head_num)
        self.head_num = head_num
        self.threshold = threshold

        self.mapping = nn.Linear(input_dim, input_dim)
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(input_dim, input_dim))
                                      for _ in range(2)])

    # create cosine similarity adj matrix
    # sem_embs: [batch_size, head_num, seq_len, d_k]
    # score_mask: [batch_size, head_num, seq_len, seq_len]
    def create_cos_adj(self, sem_embs, score_mask):
        seq_len = sem_embs.size(2)

        # calculate the cosine similarity between each words in the sentences
        a = sem_embs.unsqueeze(3)  # [batch_size, head_num, seq_len, 1, d_k]
        b = sem_embs.unsqueeze(2).repeat(1, 1, seq_len, 1, 1)  # [batch_size, head_num, seq_len, seq_len, d_k]
        cos_similarity = F.cosine_similarity(a, b, dim=-1)  # [batch_size, head_num, seq_len, seq_len]
        cos_similarity = cos_similarity * (~score_mask).float()  # mask

        # keep the value larger than threshold as the connection
        cos_adj = (cos_similarity > self.threshold).float()
        return cos_adj

    # attn = ((QW)(KW)^T)/sqrt(d)
    # query, key: [batch_size, seq_len, hidden_dim]
    # score_mask: [batch_size, head_num, seq_len, seq_len]
    def attention(self, query, key, score_mask):
        nbatches = query.size(0)
        seq_len = query.size(1)

        query, key = [l(x).view(nbatches, seq_len, self.head_num, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, head_num, seq_len, seq_len]

        scores = scores.masked_fill(score_mask, -1e10)
        p_attn = F.softmax(scores, dim=-1)  # [batch_size, head_num, seq_len, seq_len]

        b = ~score_mask[:, :, :, 0:1]
        p_attn = p_attn * b.float()  # [batch_size, head_num, seq_len, 1]
        return p_attn

    # embs: [batch_size, seq_len, input_dim]
    # score_mask: [batch_size, seq_len, seq_len]
    def forward(self, embs, score_mask):
        batch_size = embs.size(0)
        seq_len = embs.size(1)
        score_mask = score_mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)  # [batch_size, head_num, seq_len, seq_len]

        embs_mapped = self.mapping(embs)  # [batch_size, seq_len, input_dim]
        sem_embs = embs_mapped.view(batch_size, seq_len, self.head_num, self.d_k)\
                            .transpose(1, 2)  # [batch_size, head_num, seq_len, d_k]

        K_head_cosine = self.create_cos_adj(sem_embs, score_mask)   # [batch_size, head_num, seq_len, seq_len]

        # multi-head attn for embs_mapped
        attn = self.attention(embs_mapped, embs_mapped, score_mask)

        K_head_attn_cosine = K_head_cosine * attn
        return K_head_attn_cosine


class TimeWiseAspectBasedAttn(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(TimeWiseAspectBasedAttn, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(hidden_dim, hidden_dim).normal_(0, 1))
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    # context [batch_size, seq_len, hidden_dim]
    # aspect [batch_size, 1, hidden_dim]
    def forward(self, h, h_T, asp_mask, score_mask):
        h_T_ = h_T.unsqueeze(0).repeat(self.num_layers - 1, 1, 1, 1)  # [num_layers - 1, batch_size, seq_len, hidden_dim]
        a0 = torch.sum(h * h_T_, dim=-1, keepdim=True)
        a0 = F.softmax(a0, dim=0)

        h_weighted = torch.sum(a0 * h, dim=0) + h_T

        # avg pooling asp and context fearure
        # mask: [batch_size, seq_len]
        asp_wn = asp_mask.sum(dim=1, keepdim=True)  # aspect words num  [batch_size, 1]
        asp_mask = asp_mask.unsqueeze(-1).repeat(1, 1, self.hidden_dim)  # mask for h:[batch_size, seq_len, hidden_dim]

        aspect = (h_weighted * asp_mask).sum(dim=1) / asp_wn  # [batch_size, hidden_dim]
        context = h_weighted * (asp_mask == 0).float()  # [batch_size, seq_len, hidden_dim]

        # aspect based attn
        # aspect x self.W x context
        a1 = torch.matmul(aspect.unsqueeze(1), self.W)  # [batch_size, 1, hidden_dim]
        a1 = torch.matmul(a1, context.transpose(1, 2)).transpose(1, 2)  # [batch_size, seq_len, 1]
        a1 = torch.softmax(a1.masked_fill(score_mask[:, :, 0:1], -1e10), dim=1)

        # weighted and add
        context_weighted_vec = torch.sum(a1 * context, dim=1)  # [batch_size, hidden_dim]

        output = torch.cat((context_weighted_vec, aspect), dim=-1)  # [batch_size, 2 * hidden_dim]
        return output


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z, mask):
        w = self.project(z)
        w = w.masked_fill(mask[:, :, 0:1], -1e10)
        w = torch.softmax(w, dim=1)
        return w * z, w


class MultiHeadAttention(nn.Module):
    # d_model:hidden_dim，h:head_num
    def __init__(self, args, head_num, hidden_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_dim % head_num == 0

        self.d_k = int(hidden_dim // head_num)
        self.head_num = head_num
        self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(hidden_dim, hidden_dim)) for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, score_mask, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if score_mask is not None:
            scores = scores.masked_fill(score_mask, -1e9)
        b = ~score_mask[:, :, 0:1]
        p_attn = F.softmax(scores, dim=-1) * b.float()
        if dropout is not None:
            p_attn = dropout(p_attn)
        return p_attn

    def forward(self, query, key, score_mask):
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.head_num, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        attn = self.attention(query, key, score_mask, dropout=self.dropout)
        return attn

