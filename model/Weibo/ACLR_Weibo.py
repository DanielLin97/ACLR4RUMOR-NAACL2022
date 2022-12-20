import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping2class import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import GCNConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy
import random
import transformers


class TDrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(TDrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, x, data):
        edge_index = data.edge_index

        x1=copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2=copy.copy(x)
        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)
        x= scatter_mean(x, data.batch, dim=0)

        return x

class BUrumorGCN(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats):
        super(BUrumorGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats+in_feats, out_feats)

    def forward(self, x, data):
        edge_index = data.BU_edge_index

        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x

class SCL(th.nn.Module):
    def __init__(self, temperature=0.1):
        super(SCL, self).__init__()
        self.temperature = temperature

    def forward(self, inrep_1, inrep_2, label_1, label_2=None):
        inrep_1.to(device)
        inrep_2.to(device)
        bs_1 = int(inrep_1.shape[0])
        bs_2 = int(inrep_2.shape[0])

        if label_2 is None:
            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            diag = th.diag(cosine_similarity)
            cos_diag = th.diag_embed(diag)  # bs,bs

            label = th.unsqueeze(label_1, -1)
            if label.shape[0] == 1:
                cos_loss = th.zeros(1)
            else:
                for i in range(label.shape[0] - 1):
                    if i == 0:
                        label_mat = th.cat((label, label), -1)
                    else:
                        label_mat = th.cat((label_mat, label), -1)  # bs, bs
            #print(label_mat.size())
            #print(label.size())
            #exit(0)

                mid_mat_ = (label_mat.eq(label_mat.t()))
                mid_mat = mid_mat_.float()

                cosine_similarity = (cosine_similarity-cos_diag) / self.temperature  # the diag is 0
                mid_diag = th.diag_embed(th.diag(mid_mat))
                mid_mat = mid_mat - mid_diag

                cosine_similarity = cosine_similarity.masked_fill_(mid_diag.byte(), -float('inf'))  # mask the diag

                cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1) + mid_diag, 1e-10, 1e10))  # the sum of each row is 1

                cos_loss = cos_loss * mid_mat

                cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)  # bs
        else:
            if bs_1 != bs_2:
                while bs_1 < bs_2:
                    inrep_2 = inrep_2[:bs_1]
                    label_2 = label_2[:bs_1]
                    break
                while bs_2 < bs_1:
                    inrep_2_ = inrep_2
                    ra = random.randint(0, int(inrep_2_.shape[0]) - 1)
                    pad = inrep_2_[ra].unsqueeze(0)
                    lbl_pad = label_2[ra].unsqueeze(0)
                    inrep_2 = th.cat((inrep_2, pad), 0)
                    label_2 = th.cat((label_2, lbl_pad), 0)
                    bs_2 = int(inrep_2.shape[0])

            normalize_inrep_1 = F.normalize(inrep_1, p=2, dim=1)
            normalize_inrep_2 = F.normalize(inrep_2, p=2, dim=1)
            cosine_similarity = th.matmul(normalize_inrep_1, normalize_inrep_2.t())  # bs_1, bs_2

            label_1 = th.unsqueeze(label_1, -1)

            for i in range(label_1.shape[0] - 1):
                if i == 0:
                    label_1_mat = th.cat((label_1, label_1), -1)
                else:
                    label_1_mat = th.cat((label_1_mat, label_1), -1)  # bs, bs

            label_2 = th.unsqueeze(label_2, -1)

            for i in range(label_2.shape[0] - 1):
                if i == 0:
                    label_2_mat = th.cat((label_2, label_2), -1)
                else:
                    label_2_mat = th.cat((label_2_mat, label_2), -1)  # bs, bs

            mid_mat_ = (label_1_mat.t().eq(label_2_mat))
            mid_mat = mid_mat_.float()

            cosine_similarity = cosine_similarity / self.temperature
            cos_loss = th.log(th.clamp(F.softmax(cosine_similarity, dim=1), 1e-10, 1e10))
            cos_loss = cos_loss * mid_mat #find the sample with the same label
            cos_loss = th.sum(cos_loss, dim=1) / (th.sum(mid_mat, dim=1) + 1e-10)

        cos_loss = -th.mean(cos_loss, dim=0)
        
        return cos_loss


class Net(th.nn.Module):
    def __init__(self,in_feats,hid_feats,out_feats,temperature):
        super(Net, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, hid_feats, out_feats)
        self.BUrumorGCN = BUrumorGCN(in_feats, hid_feats, out_feats)
        self.fc=th.nn.Linear((out_feats+hid_feats)*2,2)
        self.scl = SCL(temperature)

    def forward(self, data, twitter_data=None):
        if twitter_data is None:
            x = data.x
            seq_len = data.seqlen

            TD_x = self.TDrumorGCN(x, data)
            BU_x = self.BUrumorGCN(x, data)
            x = th.cat((BU_x, TD_x), 1)
            x = self.fc(x)
            x = F.log_softmax(x, dim=1)
            return x
        else:
            t=twitter_data.x
            TD_t = self.TDrumorGCN(t, twitter_data)
            BU_t = self.BUrumorGCN(t, twitter_data)
            t_ = th.cat((BU_t, TD_t), 1)
            twitter_scloss = self.scl(t_,t_, twitter_data.y)
            t = self.fc(t_)
            t = F.log_softmax(t, dim=1)
            twitter_CEloss = F.nll_loss(t, twitter_data.y)

            x=data.x
            #seq_len = data.seqlen
            #x = self.embed(x, seq_len)

            TD_x = self.TDrumorGCN(x, data)
            BU_x = self.BUrumorGCN(x, data)
            x_ = th.cat((BU_x,TD_x), 1) #bs, (out_feats+hid_feats)*2
            weibocovid19_scloss = self.scl(x_,t_, data.y, twitter_data.y)

            x = self.fc(x_)
            x = F.log_softmax(x, dim=1)

            weibocovid19_CEloss = F.nll_loss(x, data.y)

            # normal_loss= 0.5*(0.7*twitter_CEloss+0.3*twitter_scloss)+ 0.5*(0.7*weibocovid19_CEloss+0.3*weibocovid19_scloss)

            x_.retain_grad()  # we need to get gradient w.r.t low-resource embeddings
            weibocovid19_CEloss.backward(retain_graph=True)
            unnormalized_noise = x_.grad.detach_()
            for p in self.parameters():
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
            norm = unnormalized_noise.norm(p=2, dim=-1)
            normalized_noise = unnormalized_noise / (norm.unsqueeze(dim=-1) + 1e-10)  # add 1e-10 to avoid Nan

            noise_norm = 1.5
            alp = 0.5

            target_noise = noise_norm * normalized_noise
            noise_x_ = x_ + target_noise
            noise_scloss = self.scl(noise_x_, t_, data.y, twitter_data.y)
            noise_CEloss = F.nll_loss(F.log_softmax(self.fc(noise_x_), dim=1), data.y)

            noise_loss = (1-alp) * noise_CEloss + alp * noise_scloss

            total_loss = (((1-alp) * twitter_CEloss + alp * twitter_scloss) + (
                        (1-alp) * weibocovid19_CEloss + alp * weibocovid19_scloss) + noise_loss) / 3

            return total_loss, x


def train_GCN(treeDic, x_test, x_train, twitter_train, TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):
    model = Net(768,512,128,0.1).to(device)  #768,512,128

    optimizer = transformers.AdamW([
        {'params': model.parameters()}
    ], lr=lr , weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        twitterdata_list = loadBiData(dataname='Twitter', treeDic=treeDic, fold_x_train=twitter_train, fold_x_test=[], TDdroprate=TDdroprate, BUdroprate=BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        twitter_loader = DataLoader(twitterdata_list, batch_size= batchsize, shuffle=True, num_workers=5)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)

        model.train()
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            for Batch_twitter in twitter_loader:
                Batch_twitter.to(device)

                loss, out_labels= model(Batch_data, Batch_twitter)
                
                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                avg_loss.append(loss.item())
                optimizer.step()
                _, pred = out_labels.max(dim=-1)
                correct = pred.eq(Batch_data.y).sum().item()
                train_acc = correct / len(Batch_data.y)
                avg_acc.append(train_acc)
                postfix = "Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc)
                tqdm_train_loader.set_postfix_str(postfix)
                batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2 = [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out = model(Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2 = evaluationclass(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_Acc_all), np.mean(temp_val_Acc1), np.mean(temp_val_Acc2), np.mean(temp_val_Prec1), np.mean(temp_val_Prec2), np.mean(temp_val_Recll1), np.mean(temp_val_Recll2), np.mean(temp_val_F1), np.mean(temp_val_F2),
                        model, 'BiGCN', dataname)
        accs = np.mean(temp_val_Acc_all)
        acc1 = np.mean(temp_val_Acc1)
        acc2 = np.mean(temp_val_Acc2)
        pre1 = np.mean(temp_val_Prec1)
        pre2 = np.mean(temp_val_Prec2)
        rec1 = np.mean(temp_val_Recll1)
        rec2 = np.mean(temp_val_Recll2)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        if early_stopping.early_stop:
            print("Early stopping")
            accs = early_stopping.accs
            acc1 = early_stopping.acc1
            acc2 = early_stopping.acc2
            pre1 = early_stopping.pre1
            pre2 = early_stopping.pre2
            rec1 = early_stopping.rec1
            rec2 = early_stopping.rec2
            F1 = early_stopping.F1
            F2 = early_stopping.F2
            break
    return train_losses , val_losses ,train_accs, val_accs, accs, acc1, pre1, rec1, F1, acc2, pre2, rec2, F2

lr=0.0001
weight_decay=1e-4
patience=30
n_epochs=200
batchsize=32
TDdroprate=0.2
BUdroprate=0.2
datasetname=sys.argv[1] #"Twitter"、"Weibo"
iterations=int(sys.argv[2])
model="GCN"
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = []
ACC1, ACC2, PRE1, PRE2, REC1, REC2, F1, F2 = [], [], [], [], [], [], [], []

for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test,  fold1_x_train,  \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test,fold4_x_train, twitter_train = load5foldData(datasetname)
    treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs, accs_0, acc1_0, pre1_0, rec1_0,  F1_0, acc2_0, pre2_0, rec2_0, F2_0 = train_GCN(treeDic,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train, twitter_train,
                                                                                               TDdroprate,BUdroprate,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs, accs_1, acc1_1, pre1_1, rec1_1, F1_1, acc2_1, pre2_1, rec2_1, F2_1 = train_GCN(treeDic,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train, twitter_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs, accs_2, acc1_2, pre1_2, rec1_2, F1_2, acc2_2, pre2_2, rec2_2, F2_2 = train_GCN(treeDic,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train, twitter_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs, accs_3, acc1_3, pre1_3, rec1_3, F1_3, acc2_3, pre2_3, rec2_3, F2_3 = train_GCN(treeDic,
                                                                                               fold3_x_test,
                                                                                               fold3_x_train, twitter_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs, accs_4, acc1_4, pre1_4, rec1_4, F1_4, acc2_4, pre2_4, rec2_4, F2_4 = train_GCN(treeDic,
                                                                                               fold4_x_test,
                                                                                               fold4_x_train, twitter_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    test_accs.append((accs_0+accs_1+accs_2+accs_3+accs_4)/5)
    ACC1.append((acc1_0 + acc1_1 + acc1_2 + acc1_3 + acc1_4) / 5)
    ACC2.append((acc2_0 + acc2_1 + acc2_2 + acc2_3 + acc2_4) / 5)
    PRE1.append((pre1_0 + pre1_1 + pre1_2 + pre1_3 + pre1_4) / 5)
    PRE2.append((pre2_0 + pre2_1 + pre2_2 + pre2_3 + pre2_4) / 5)
    REC1.append((rec1_0 + rec1_1 + rec1_2 + rec1_3 + rec1_4) / 5)
    REC2.append((rec2_0 + rec2_1 + rec2_2 + rec2_3 + rec2_4) / 5)
    F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    F2.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
print("Twitter:|Total_Test_ Accuracy: {:.4f}|acc1: {:.4f}|acc2: {:.4f}|pre1: {:.4f}|pre2: {:.4f}"
                  "|rec1: {:.4f}|rec2: {:.4f}|F1: {:.4f}|F2: {:.4f}".format(sum(test_accs) / iterations, sum(ACC1) / iterations,
                                                                            sum(ACC2) / iterations, sum(PRE1) / iterations, sum(PRE2) /iterations,
                                                                            sum(REC1) / iterations, sum(REC2) / iterations, sum(F1) / iterations, sum(F2) / iterations))


