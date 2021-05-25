import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gcn import GCNClassifier


class GCNTrainer(object):
    def __init__(self, args, emb_matrix=None):
        self.args = args
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(args, emb_matrix=emb_matrix).to(self.args.device)

        if args.emb_type == 'bert':
            bert_model = self.model.gcn_model.bert
            bert_params_dict = list(map(id, bert_model.parameters()))
            base_params = filter(lambda p: id(p) not in bert_params_dict, self.model.parameters())
            self.parameters = [
                {"params": base_params},
                {"params": bert_model.parameters(), "lr": args.bert_lr},
            ]
        else:
            self.parameters = self.model.parameters()

        self.optimizer = torch.optim.Adam(
                self.parameters, lr=args.lr, weight_decay=args.l2reg)

    # load model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.args = checkpoint['config']

    # save model
    def save(self, filename):
        params = {
                'model': self.model.state_dict(),
                'config': self.args,
                }
        try:
            torch.save(params, filename, _use_new_zipfile_serialization=False)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def update(self, batch):
        if self.args.emb_type == "glove":
            inputs = batch[0:9]
        elif self.args.emb_type == "bert":
            inputs = batch[0:11]
        label = batch[-1]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, gcn_outputs, h0, h1= self.model(inputs)

        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        
        # backward
        loss.backward()
        self.optimizer.step()
        return loss.data, acc

    def predict(self, batch):
        if self.args.emb_type == "glove":
            inputs = batch[0:9]
        elif self.args.emb_type == "bert":
            inputs = batch[0:11]
        label = batch[-1]

        # forward
        self.model.eval()
        logits, gcn_outputs, h0, h1 = self.model(inputs)

        loss = F.cross_entropy(logits, label, reduction='mean')
        corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
        acc = 100.0 * np.float(corrects) / label.size()[0]
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        predprob = F.softmax(logits, dim=1).data.cpu().numpy().tolist()

        return loss.data, acc, predictions, \
               label.data.cpu().numpy().tolist(), predprob, \
               gcn_outputs.data.cpu().numpy()
