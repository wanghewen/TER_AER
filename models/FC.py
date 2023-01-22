from torch.nn import ModuleList, Linear, BatchNorm1d
from torch import nn
import torch
import torch.nn.functional as F
# pl.seed_everything(12)
from torch.utils.data import TensorDataset

from models.Base import BaseModel


class MLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()
        self.dropout = dropout

        self.lins = ModuleList([Linear(in_channels, hidden_channels)])
        self.bns = ModuleList([BatchNorm1d(hidden_channels)])
        for _ in range(num_layers - 1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

    def forward(self, x):
        for lin, bn in zip(self.lins[:-1], self.bns):
            x = lin(x).relu_()
            x = bn(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


class FCBaseline(BaseModel):

    def __init__(self, in_nfeats, labelsize, n_layers, n_hidden, dropout):
        super().__init__(n_classes=labelsize)
        #         self.fc = nn.Linear(embdim, labelsize)
        # self.train_dataset = TensorDataset(torch.FloatTensor(train_embeds), torch.LongTensor(train_labels))
        # self.val_dataset = TensorDataset(torch.FloatTensor(val_embeds), torch.LongTensor(val_labels))
        # self.test_dataset = TensorDataset(torch.FloatTensor(test_embeds), torch.LongTensor(test_labels))
        self.fc = MLP(in_nfeats, labelsize, hidden_channels=n_hidden,
                      num_layers=n_layers, dropout=dropout)
        #         self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn = nn.BCEWithLogitsLoss()



    def forward(self, batch):
        # in lightning, forward defines the prediction/inference actions
        x, labels = batch[0], batch[1]
        preds = self.fc(x)
        #         pred_labels = torch.argmax(preds, dim=1)
        preds = torch.sigmoid(preds)
        # pred_labels = (preds > 0.5).float()
        return labels, preds

    # def _step(self, batch, batch_idx, dataset):
    def _step(self, batch):
        # training_step defined the train loop. It is independent of forward
        # index = batch
        # x, labels = dataset[index][0], dataset[index][1]
        x, labels = batch[0], batch[1]
        x, labels = x.to(self.device), labels.to(self.device)
        # x = x.view(x.size(0), -1)
        preds = self.fc(x)
        loss = self.loss_fn(preds, labels.float())
        preds = torch.sigmoid(preds)
        #         pred_labels = torch.argmax(preds, dim=1)
        pred_labels = (preds > 0.5).float()
        return {'loss': loss, "preds": preds, "pred_labels": pred_labels, "labels": labels}

    #         self.log('train_loss', loss)
    #         self.log('train_accuracy_step', self.train_acc(pred_labels, labels), on_step=True, on_epoch=False)
    #         self.log('train_macro_f1_step', self.train_macro_f1(preds, labels), on_step=True, on_epoch=False)
    #         return loss

    def training_step(self, batch, batch_idx):
        # return self._step(batch, batch_idx, self.train_dataset)
        return self._step(batch)

    def training_epoch_end(self, output):
        # log epoch metric
        preds = torch.cat([x['preds'] for x in output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).mean()

        self.log_result(preds, pred_labels, labels, loss, dataset="train")
        #         self.log('train_accuracy_epoch', self.train_acc(pred_labels, labels))
        #         self.log('train_macro_f1_epoch', self.train_macro_f1(preds, labels))
        #         self.train_acc.reset()
        #         self.train_macro_f1.reset()
        pass

    def validation_step(self, batch, batch_idx):
        return self._step(batch)

    def test_step(self, batch, batch_idx):
        return self._step(batch)

    def validation_epoch_end(self, output):
        # log epoch metric
        preds = torch.cat([x['preds'] for x in output]).detach()
        #         import pdb;pdb.set_trace()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).mean()
        #         print(preds.shape, labels.shape)
        #         import pdb;pdb.set_trace()
        # self.hhh = pred_labels

        self.log_result(preds, pred_labels, labels, loss, dataset="val")

    def test_epoch_end(self, output):
        preds = torch.cat([x['preds'] for x in output]).detach()
        pred_labels = torch.cat([x['pred_labels'] for x in output]).detach()
        labels = torch.cat([x['labels'] for x in output]).detach()
        loss = torch.stack([x['loss'] for x in output]).mean()

        self.log_result(preds, pred_labels, labels, loss, dataset="test")

    #         self.log('val_accuracy_epoch', self.valid_acc(pred_labels, labels))
    #         self.log('val_macro_f1_epoch', self.valid_macro_f1(preds, labels))
    #         self.valid_acc.reset()
    #         self.valid_macro_f1.reset()

