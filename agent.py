import os
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import networkx as nx
from sklearn.metrics import accuracy_score

from utils import *
from dataset import get_eigh

import matplotlib.pyplot as plt

from model.sgc import SGC
from model.gcn import GCN
from model.appnp import APPNP
from model.chebnet import ChebNet
from model.chebnetII import ChebNetII
from model.bernnet import BernNet
from model.gprgnn import GPRGNN
from model.gat import GAT
from mine.models.mine import Mine
class GraphAgent:
    def __init__(self, args, data):
        self.args = args
        self.data = data
        self.n_syn = int(len(data.idx_train) * args.reduction_rate)
        self.n_middle = int(self.n_syn*0.5)
        self.d = (data.x_train).shape[1]
        self.num_classes = data.num_classes
        self.syn_class_indices = {}
        self.syn_class_indices_middle = {}
        self.class_dict = None
        self.x_middle = nn.Parameter(torch.FloatTensor(int(self.n_syn*0.5), self.d).cuda())
        self.x_syn = nn.Parameter(torch.FloatTensor(self.n_syn, self.d).cuda())
        self.eigenvecs_syn = nn.Parameter(
            torch.FloatTensor(self.n_syn, args.eigen_k).cuda()
        )
        self.eigenvecs_syn_middle = nn.Parameter(
            torch.FloatTensor(int(self.n_syn*0.5), int(args.eigen_k*0.5)).cuda()
        )
        y_full = data.y_full
        idx_train = data.idx_train
        self.y_syn = torch.LongTensor(self.generate_labels_syn(y_full[idx_train], args.reduction_rate)).cuda()
        self.train_label=y_full[idx_train]
        self.y_middle = torch.LongTensor(self.generate_labels_middle(y_full[idx_train], args.reduction_rate*0.5)).cuda()
        init_syn_feat = self.get_init_syn_feat(dataset=args.dataset, reduction_rate=args.reduction_rate,
                                               expID=args.expID)
        init_syn_eigenvecs = self.get_init_syn_eigenvecs(self.n_syn, self.num_classes)
        init_syn_eigenvecs = init_syn_eigenvecs[:, :args.eigen_k]
        print(args.reduction_rate)
        print(init_syn_feat.shape)
        print(init_syn_eigenvecs.shape)
        self.reset_parameters(init_syn_feat, init_syn_eigenvecs)

    def reset_parameters(self, init_syn_feat, init_syn_eigenvecs):
        self.x_syn.data.copy_(init_syn_feat)
        self.eigenvecs_syn.data.copy_(init_syn_eigenvecs)

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.y_syn.cpu().numpy())

        for c in range(data.num_classes):
            tmp = self.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]

        return features

    def retrieve_class(self, c, num=256):
        y_train = self.data.y_train.cpu().numpy()
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.data.num_classes):
                self.class_dict['class_%s' % i] = (y_train == i)
        idx = np.arange(len(self.data.idx_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def train(self, eigenvals_syn, co_x_trans_real, embed_mean_real):
        args = self.args
        data = self.data
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)

        optimizer_feat = torch.optim.Adam(
            [self.x_syn], lr=args.lr_feat
        )
        optimizer_eigenvec = torch.optim.Adam(
            [self.eigenvecs_syn], lr=args.lr_eigenvec
        )

        for ep in range(args.epoch):
            loss = 0.0
            x_syn = self.x_syn
            eigenvecs_syn = self.eigenvecs_syn
            # eigenbasis match
            co_x_trans_syn = get_subspace_covariance_matrix(eigenvecs=eigenvecs_syn, x=x_syn)  # kdd
            eigen_match_loss = F.mse_loss(co_x_trans_syn, co_x_trans_real)
            loss += args.alpha * eigen_match_loss
            # class loss
            embed_sum_syn = get_embed_sum(eigenvals=eigenvals_syn, eigenvecs=eigenvecs_syn, x=x_syn)
            embed_mean_syn = get_embed_mean(embed_sum=embed_sum_syn, label=self.y_syn)  # cd
            cov_embed = embed_mean_real @ embed_mean_syn.T
            iden = torch.eye(data.num_classes).cuda()
            class_loss = F.mse_loss(cov_embed, iden)
            loss += args.beta * class_loss
            # orthog_norm
            orthog_syn = eigenvecs_syn.T @ eigenvecs_syn
            iden = torch.eye(args.eigen_k).cuda()
            orthog_norm = F.mse_loss(orthog_syn, iden)
            loss += args.gamma * orthog_norm

            if (ep == 0) or (ep == (args.epoch - 1)):
                print(f"epoch: {ep}")
                print(f"eigen_match_loss: {eigen_match_loss}")
                print(f"args.alpha * eigen_match_loss: {args.alpha * eigen_match_loss}")

                print(f"class_loss: {class_loss}")
                print(f"args.beta * class_loss: {args.beta * class_loss}")

                print(f"orthog_norm: {orthog_norm}")
                print(f"args.gamma * orthog_norm: {args.gamma * orthog_norm}")

            optimizer_eigenvec.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()

            # update U:
            if ep % (args.e1 + args.e2) < args.e1:
                optimizer_eigenvec.step()
            else:
                optimizer_feat.step()

        x_syn, y_syn = self.x_syn.detach(), self.y_syn
        eigenvecs_syn = self.eigenvecs_syn.detach()

        acc = self.test_with_val(
            x_syn,
            eigenvals_syn,
            eigenvecs_syn,
            y_syn,
            verbose=False
        )

        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        torch.save(
            eigenvals_syn,
            f"{dir}/eigenvals_syn_{args.expID}.pt",
        )
        torch.save(
            eigenvecs_syn,
            f"{dir}/eigenvecs_syn_{args.expID}.pt",
        )
        torch.save(
            x_syn, f"{dir}/feat_{args.expID}.pt"
        )

        return acc
    def train_middle(self, eigenvals_syn, co_x_trans_real, embed_mean_real):
        args = self.args
        data = self.data
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)

        optimizer_feat = torch.optim.Adam(
            [self.x_syn], lr=args.lr_feat
        )
        optimizer_eigenvec = torch.optim.Adam(
            [self.eigenvecs_syn], lr=args.lr_eigenvec
        )

        for ep in range(args.epoch):
            loss = 0.0
            x_syn = self.x_syn
            eigenvecs_syn = self.eigenvecs_syn
            # eigenbasis match
            co_x_trans_syn = get_subspace_covariance_matrix(eigenvecs=eigenvecs_syn, x=x_syn)  # kdd
            eigen_match_loss = F.mse_loss(co_x_trans_syn, co_x_trans_real)
            loss += args.alpha * eigen_match_loss
            # class loss
            embed_sum_syn = get_embed_sum(eigenvals=eigenvals_syn, eigenvecs=eigenvecs_syn, x=x_syn)
            embed_mean_syn = get_embed_mean(embed_sum=embed_sum_syn, label=self.y_syn)  # cd
            cov_embed = embed_mean_real @ embed_mean_syn.T
            iden = torch.eye(data.num_classes).cuda()
            class_loss = F.mse_loss(cov_embed, iden)
            loss += args.beta * class_loss
            # orthog_norm
            orthog_syn = eigenvecs_syn.T @ eigenvecs_syn
            iden = torch.eye(args.eigen_k).cuda()
            orthog_norm = F.mse_loss(orthog_syn, iden)
            loss += args.gamma * orthog_norm

            if (ep == 0) or (ep == (args.epoch - 1)):
                print(f"epoch: {ep}")
                print(f"eigen_match_loss: {eigen_match_loss}")
                print(f"args.alpha * eigen_match_loss: {args.alpha * eigen_match_loss}")

                print(f"class_loss: {class_loss}")
                print(f"args.beta * class_loss: {args.beta * class_loss}")

                print(f"orthog_norm: {orthog_norm}")
                print(f"args.gamma * orthog_norm: {args.gamma * orthog_norm}")

            optimizer_eigenvec.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()

            # update U:
            if ep % (args.e1 + args.e2) < args.e1:
                optimizer_eigenvec.step()
            else:
                optimizer_feat.step()
        x_syn, y_syn = self.x_syn.detach(), self.y_syn
        eigenvecs_syn = self.eigenvecs_syn.detach()
        rat_list=[1.0]
        for rat in rat_list:
           
            acc = self.test_with_val(
                x_syn,
                eigenvals_syn,
                eigenvecs_syn,
                y_syn,
                verbose=False,
                ipc=rat
            )

        args.reduction_rate=args.reduction_rate*2
        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        torch.save(
            eigenvals_syn,
            f"{dir}/eigenvals_syn_middle.pt",
        )
        torch.save(
            eigenvecs_syn,
            f"{dir}/eigenvecs_syn_middle.pt",
        )
        torch.save(
            x_syn, f"{dir}/feat_middle.pt"
        )

        return acc
    def transfer(self,x_middle):
        for c in range(len(self.syn_class_indices)):
            [start, end] = self.syn_class_indices[c]
            [start_middle, end_middle] = self.syn_class_indices_middle[c]
            length = end_middle - start_middle
            self.x_syn.data[start:start + length, :] = x_middle[start_middle:end_middle, :]
    def mask_calculate(self,stop_mask):
        for c, (start_middle, end_middle) in self.syn_class_indices_middle.items():
            length = end_middle - start_middle

            
            random_values = torch.bernoulli(torch.full((length, 1), 0.5))

            if not torch.any(random_values == 1):
                random_index = torch.randint(0, length, (1,))
                random_values[random_index] = 1

            stop_mask[start_middle:end_middle, :] = random_values
        return stop_mask
    def mask_calculate_top(self,stop_mask):
        for c, (start, end) in self.syn_class_indices.items():
            length = end - start
            (start_middle,end_middle)=self.syn_class_indices_middle[c]
            len2=end_middle-start_middle

            random_values = torch.bernoulli(torch.full((length, 1), 0.5))
            random_values[0:len2,:] = torch.ones(len2,1)
          
            if not torch.any(random_values == 1):
                random_index = torch.randint(0, length, (1,))
                random_values[random_index] = 1
            random_values=random_values.to('cuda')
            
            stop_mask[start:end, :] = random_values
        return stop_mask
    def mask_increase(self, stop_mask):
        for c, (start, end) in self.syn_class_indices.items():
            length = end - start
            random_values = torch.bernoulli(torch.full((length, 1), 0.5)).int()
            value2 = stop_mask[start:end,:].int()
            random_values = random_values.to('cuda')
            value2 = value2.to('cuda')
            random_values = torch.logical_or(random_values,value2).float()
            if not torch.any(random_values == 1):
                random_index = torch.randint(0, length, (1,))
                random_values[random_index] = 1

            stop_mask[start:end, :] = random_values
        return stop_mask
    def mask_decrease(self, stop_mask):
        for c, (start_middle, end_middle) in self.syn_class_indices_middle.items():
            length = end_middle - start_middle

            random_values = torch.bernoulli(torch.full((length, 1), 0.5))
            value2 = stop_mask[start_middle:end_middle,:]
            random_values = random_values.to('cuda')
            random_values = random_values * value2

            if not torch.any(random_values == 1):
                random_index = torch.randint(0, length, (1,))
                random_values[random_index] = 1

            stop_mask[start_middle:end_middle, :] = random_values
        return stop_mask
    def train_bottom(self, eigenvals_syn, co_x_trans_real, embed_mean_real):
        args = self.args
        data = self.data
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)
        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        self.x_middle.data=torch.load(f"{dir}/feat_middle.pt")
        self.eigenvecs_syn_middle.data=torch.load(f"{dir}/eigenvecs_syn_middle.pt")
        #
       # self.x_syn.data= self.transfer(x_middle)
      #  self.eigenvecs_syn=self.transfer(eigenvecs_syn_middle)
        (n,d)=self.x_middle.shape
        optimizer_feat = torch.optim.Adam(
            [self.x_middle], lr=args.lr_feat
        )
        optimizer_eigenvec = torch.optim.Adam(
            [self.eigenvecs_syn_middle], lr=args.lr_eigenvec
        )
        stop_mask = torch.zeros(n, 1)
        stop_mask = self.mask_calculate(stop_mask)
        continue_mask = torch.ones(n,1)
        continue_mask = continue_mask-stop_mask
        stop_mask = stop_mask.to('cuda')
        continue_mask = continue_mask.to('cuda')
        zero_number=0.0
        mine = Mine(self.d)
        for ep in range(args.epoch_bottom):
            loss = 0.0
            x_syn = self.x_middle
            eigenvecs_syn = self.eigenvecs_syn_middle
            # eigenbasis match
            co_x_trans_syn = get_subspace_covariance_matrix(eigenvecs=eigenvecs_syn, x=x_syn)  # kdd
            eigen_match_loss = F.mse_loss(co_x_trans_syn, co_x_trans_real)
            loss += args.alpha * eigen_match_loss
            # class loss
            embed_sum_syn = get_embed_sum(eigenvals=eigenvals_syn, eigenvecs=eigenvecs_syn, x=x_syn)
            embed_mean_syn = get_embed_mean(embed_sum=embed_sum_syn, label=self.y_middle)  # cd
            cov_embed = embed_mean_real @ embed_mean_syn.T
            iden = torch.eye(data.num_classes).cuda()
            class_loss = F.mse_loss(cov_embed, iden)
            loss += args.beta * class_loss
            # orthog_norm
            orthog_syn = eigenvecs_syn.T @ eigenvecs_syn
            iden = torch.eye(args.eigen_k_middle).cuda()
            orthog_norm = F.mse_loss(orthog_syn, iden)
            loss += args.gamma * orthog_norm
            if zero_number<0.4 and self.args.use_mine:
                x_selected = x_syn[stop_mask.squeeze() == 1] 
                y_selected = self.y_middle[stop_mask.squeeze() == 1]  
                mean_selected = self.compute_class_means(x_selected,y_selected)
                mean_syn = self.compute_class_means(x_syn,self.y_middle)
                ib_beta = 1e-30
                mi = mine.optimize(mean_syn, mean_selected, iters = 20, batch_size=self.num_classes)
                loss -= ib_beta*mi.detach()
                print(mi)
            if (ep == 0) or (ep == (args.epoch - 1)):
                print(f"epoch: {ep}")
                print(f"eigen_match_loss: {eigen_match_loss}")
                print(f"args.alpha * eigen_match_loss: {args.alpha * eigen_match_loss}")

                print(f"class_loss: {class_loss}")
                print(f"args.beta * class_loss: {args.beta * class_loss}")

                print(f"orthog_norm: {orthog_norm}")
                print(f"args.gamma * orthog_norm: {args.gamma * orthog_norm}")
            feat_origin=x_syn.data
            feat_origin=feat_origin.to('cuda')
            optimizer_eigenvec.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()

            # update U:
            # if ep % (args.e1 + args.e2) < args.e1:
            #     optimizer_eigenvec.step()
            # else:
            optimizer_feat.step()
            self.x_middle.data = continue_mask * feat_origin+ stop_mask * x_syn.data
            stop_mask = self.mask_decrease(stop_mask)
            continue_mask = torch.ones(stop_mask.shape).to('cuda')-stop_mask
            zero_number=int(torch.sum(stop_mask==0))
            # print(f'zero ratio:{zero_number/n}')
        x_syn, y_syn = self.x_middle.detach(), self.y_middle
        eigenvecs_syn = self.eigenvecs_syn_middle.detach()
        rat=[0.5,1.0]
        for ipc in rat:
            print(f'Randomly selected {ipc*0.5} rate from the largest condensed graph for testing')
          #  print(ipc)
            acc = self.test_with_val_middle(
                x_syn,
                eigenvals_syn,
                eigenvecs_syn,
                y_syn,
                verbose=False,
                ipc=ipc
            )

        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        torch.save(
            eigenvals_syn,
            f"{dir}/eigenvals_syn_bottom.pt",
        )
        torch.save(
            eigenvecs_syn,
            f"{dir}/eigenvecs_syn_bottom.pt",
        )
        torch.save(
            x_syn, f"{dir}/feat_bottom.pt"
        )

        return acc

    def train_top(self, eigenvals_syn, co_x_trans_real, embed_mean_real):
        args = self.args
        data = self.data
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)
        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        x_middle = torch.load(f"{dir}/feat_middle.pt")
      #  eigenvecs_syn_middle.data = torch.load(f"{dir}/eigenvecs_syn_middle.pt")
        #
        self.transfer(x_middle)
       # self.eigenvecs_syn=self.transfer(eigenvecs_syn_middle)
        optimizer_feat = torch.optim.Adam(
            [self.x_syn], lr=args.lr_feat
        )
        optimizer_eigenvec = torch.optim.Adam(
            [self.eigenvecs_syn], lr=args.lr_eigenvec
        )
        (n,d) = self.x_syn.shape
        stop_mask = torch.zeros((n,1)).to('cuda')
        stop_mask = self.mask_calculate_top(stop_mask)
        continue_mask =torch.ones((n,1)).to('cuda')
        continue_mask = continue_mask - stop_mask
        zero_number = 1
        mine = Mine(self.d)
        for ep in range(args.epoch):
            loss = 0.0
            x_syn = self.x_syn
            eigenvecs_syn = self.eigenvecs_syn
            co_x_trans_syn = get_subspace_covariance_matrix(eigenvecs=eigenvecs_syn, x=x_syn)  # kdd
            eigen_match_loss = F.mse_loss(co_x_trans_syn, co_x_trans_real)
            loss += args.alpha * eigen_match_loss
            embed_sum_syn = get_embed_sum(eigenvals=eigenvals_syn, eigenvecs=eigenvecs_syn, x=x_syn)
            embed_mean_syn = get_embed_mean(embed_sum=embed_sum_syn, label=self.y_syn)  # cd
            cov_embed = embed_mean_real @ embed_mean_syn.T
            iden = torch.eye(data.num_classes).cuda()
            class_loss = F.mse_loss(cov_embed, iden)
            loss += args.beta * class_loss
            orthog_syn = eigenvecs_syn.T @ eigenvecs_syn
            iden = torch.eye(args.eigen_k).cuda()
            orthog_norm = F.mse_loss(orthog_syn, iden)
            loss += args.gamma * orthog_norm
            if zero_number>0.3 and self.args.use_mine:
                x_selected = x_syn[stop_mask.squeeze() == 1] 
                y_selected = self.y_syn[stop_mask.squeeze() == 1]  
                mean_selected = self.compute_class_means(x_selected,y_selected)
                mean_syn = self.compute_class_means(x_syn,self.y_syn)
                ib_beta = 1e-30
                mi = mine.optimize(mean_syn, mean_selected, iters = 20, batch_size=self.num_classes)
                loss -= ib_beta*mi.detach()
                # print(mi)
            if (ep == 0) or (ep == (args.epoch - 1)):
                print(f"epoch: {ep}")
                print(f"eigen_match_loss: {eigen_match_loss}")
                print(f"args.alpha * eigen_match_loss: {args.alpha * eigen_match_loss}")

                print(f"class_loss: {class_loss}")
                print(f"args.beta * class_loss: {args.beta * class_loss}")

                print(f"orthog_norm: {orthog_norm}")
                print(f"args.gamma * orthog_norm: {args.gamma * orthog_norm}")
            feat_origin = x_syn.data
            feat_origin = feat_origin.to('cuda')
            optimizer_eigenvec.zero_grad()
            optimizer_feat.zero_grad()
            loss.backward()

            # update U:
            if ep % (args.e1 + args.e2) < args.e1:
                optimizer_eigenvec.step()
            else:
                optimizer_feat.step()
                self.x_syn.data = continue_mask * feat_origin + stop_mask * x_syn.data
                stop_mask = self.mask_increase(stop_mask)
                continue_mask = torch.ones(stop_mask.shape).to('cuda') - stop_mask
                zero_number = int(torch.sum(stop_mask == 0))
               
        x_syn, y_syn = self.x_syn.detach(), self.y_syn
        eigenvecs_syn = self.eigenvecs_syn.detach()
        rat = [0.75,1.0]
        for ipc in rat:
            print(f'Randomly selected {ipc} rate from the largest condensed graph for testing')
            acc = self.test_with_val(
                x_syn,
                eigenvals_syn,
                eigenvecs_syn,
                y_syn,
                verbose=False,
                ipc=ipc
            )


        dir = f"./saved_ours/{args.dataset}-{args.reduction_rate}"
        if not os.path.isdir(dir):
            os.makedirs(dir)

        torch.save(
            eigenvals_syn,
            f"{dir}/eigenvals_syn_top.pt",
        )
        torch.save(
            eigenvecs_syn,
            f"{dir}/eigenvecs_syn_top.pt",
        )
        torch.save(
            x_syn, f"{dir}/feat_top.pt"
        )

        return acc

    def test_with_val(
            self,
            x_syn,
            eigenvals_syn,
            eigenvecs_syn,
            y_syn,
            verbose=False,
            ipc=1.0
    ):
        args = self.args
        data = self.data
        evaluate_gnn = args.evaluate_gnn
        # 计算拉普拉斯矩阵并重新回到邻接矩阵
        L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T
        if evaluate_gnn == "MLP":
            adj_syn = torch.eye(self.n_syn).cuda()
        else:
            adj_syn = torch.eye(self.n_syn).cuda() - L_syn
       # adj_syn = torch.eye(self.n_syn).cuda()
        indices = []
        from collections import Counter

        counter = Counter(self.train_label.cpu().numpy())

        for label in range(data.num_classes):
            ind = self.syn_class_indices[label]
            all = np.arange(ind[0], ind[1])
            n_class = counter[label]
            num = max(int(n_class * self.args.reduction_rate * ipc), 1)
            # print(all)
            selected = random.sample(list(all), num)
            for x in selected:
                indices.append(x)

        x_syn = x_syn[indices]
        y_syn = y_syn[indices]
        adj_syn = adj_syn[indices, :][:, indices]
        print(x_syn.shape, y_syn.shape, adj_syn.shape)

        if evaluate_gnn == "SGC":
            model = SGC(
                num_features=self.d,
                num_classes=data.num_classes,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
            ).cuda()
        elif evaluate_gnn == "GCN":
            model = GCN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout
            ).cuda()
        elif evaluate_gnn == "MLP":
            model = GCN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout
            ).cuda()
        elif evaluate_gnn == "GAT":
            model = GAT(
                nfeat=self.d,
                nclass=data.num_classes,
                nhid=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
                device='cuda0'
            ).cuda()
        elif evaluate_gnn == "ChebNet":
            model = ChebNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                k=args.k,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
            ).cuda()
        elif evaluate_gnn == "APPNP":
            model = APPNP(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
                alpha=0.1,
            ).cuda()
        elif evaluate_gnn == "ChebNetII":
            model = ChebNetII(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate
            ).cuda()
        elif evaluate_gnn == "BernNet":
            model = BernNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()
        elif evaluate_gnn == "GPRGNN":
            model = GPRGNN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()

        model.cuda()
        model.fit_with_val(
            x_syn,
            y_syn,
            adj_syn,
            data,
            args.epoch_gnn,
            verbose=verbose
        )

        model.eval()

        # Full graph
        idx_test = data.idx_test
        x_full = data.x_full
        y_full = data.y_full
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)

        y_test = (y_full[idx_test]).cpu().numpy()
        output = model.predict(x_full, adj_full)
        loss_test = F.nll_loss(output[idx_test], y_full[idx_test])

        pred = output.max(1)[1].cpu().numpy()
        acc_test = accuracy_score(y_test, pred[idx_test])

        print(
            f"(Test set results: loss= {loss_test.item():.4f}, accuracy= {acc_test:.4f}\n"
        )

        return acc_test

    def test_with_val_middle(
            self,
            x_syn,
            eigenvals_syn,
            eigenvecs_syn,
            y_syn,
            verbose=False,
            ipc=1.0
    ):
        args = self.args
        data = self.data
        evaluate_gnn = args.evaluate_gnn
        L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T
        if evaluate_gnn == "MLP":
            adj_syn = torch.eye(self.n_middle).cuda()
        else:
            adj_syn = torch.eye(self.n_middle).cuda() - L_syn
        #adj_syn = torch.eye(self.n_middle).cuda()

        from collections import Counter

        counter = Counter(self.train_label.cpu().numpy())

        indices = []
        for label in range(data.num_classes):
            ind = self.syn_class_indices_middle[label]
            all = np.arange(ind[0], ind[1])
            n_class=counter[label]
            num=max(int( n_class * self.args.reduction_rate * 0.5 * ipc ), 1)

            # print(all)
            selected = random.sample(list(all), num)
            for x in selected:
                indices.append(x)

        x_syn = x_syn[indices]
        y_syn = y_syn[indices]
        adj_syn = adj_syn[indices, :][:, indices]
        print(x_syn.shape, y_syn.shape, adj_syn.shape)

        if evaluate_gnn == "SGC":
            model = SGC(
                num_features=self.d,
                num_classes=data.num_classes,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
            ).cuda()
        elif evaluate_gnn == "GCN":
            model = GCN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout
            ).cuda()
        elif evaluate_gnn == "MLP":
            model = GCN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout
            ).cuda()
        elif evaluate_gnn == "GAT":
            model = GAT(
                nfeat=self.d,
                nclass=data.num_classes,
                nhid=args.hidden_dim,
                nlayers=args.nlayers,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
                device='cuda0'
            ).cuda()
        elif evaluate_gnn == "ChebNet":
            model = ChebNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                nlayers=args.nlayers,
                k=args.k,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
            ).cuda()
        elif evaluate_gnn == "APPNP":
            model = APPNP(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                weight_decay=args.wd_gnn,
                dropout=args.dropout,
                alpha=0.1,
            ).cuda()
        elif evaluate_gnn == "ChebNetII":
            model = ChebNetII(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate
            ).cuda()
        elif evaluate_gnn == "BernNet":
            model = BernNet(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()
        elif evaluate_gnn == "GPRGNN":
            model = GPRGNN(
                num_features=self.d,
                num_classes=data.num_classes,
                hidden_dim=args.hidden_dim,
                k=args.k,
                lr=args.lr_gnn,
                lr_conv=args.lr_conv,
                weight_decay=args.wd_gnn,
                wd_conv=args.wd_conv,
                dropout=args.dropout,
                dprate=args.dprate,
            ).cuda()

        model.cuda()
        model.fit_with_val(
            x_syn,
            y_syn,
            adj_syn,
            data,
            args.epoch_gnn,
            verbose=verbose
        )

        model.eval()

        # Full graph
        idx_test = data.idx_test
        x_full = data.x_full
        y_full = data.y_full
        adj_full = data.adj_full
        adj_full = normalize_adj_to_sparse_tensor(adj_full)

        y_test = (y_full[idx_test]).cpu().numpy()
        output = model.predict(x_full, adj_full)
        loss_test = F.nll_loss(output[idx_test], y_full[idx_test])

        pred = output.max(1)[1].cpu().numpy()
        acc_test = accuracy_score(y_test, pred[idx_test])

        print(
            f"(Test set results: loss= {loss_test.item():.4f}, accuracy= {acc_test:.4f}\n"
        )

        return acc_test

    def get_eigenspace_embed(self, eigen_vecs, x):
        eigen_vecs = eigen_vecs.unsqueeze(2)  # k * n * 1
        eigen_vecs_t = eigen_vecs.permute(0, 2, 1)  # k * 1 * n
        eigenspace = torch.bmm(eigen_vecs, eigen_vecs_t)  # knn
        embed = torch.matmul(eigenspace, x)  # knn*nd=knd
        return embed

    def get_real_embed(self, k, L, x):
        filtered_x = x

        emb_list = []
        for i in range(k):
            filtered_x = L @ filtered_x
            emb_list.append(filtered_x)

        embed = torch.stack(emb_list, dim=0)
        return embed

    def get_syn_embed(self, k, eigenvals, eigen_vecs, x):
        trans_x = eigen_vecs @ x
        filtered_x = trans_x

        emb_list = []
        for i in range(k):
            filtered_x = torch.diag(eigenvals) @ filtered_x
            emb_list.append(eigen_vecs.T @ filtered_x)

        embed = torch.stack(emb_list, dim=0)
        return embed

    def get_init_syn_feat(self, dataset, reduction_rate, expID):
        init_syn_x = torch.load(f"./initial_feat/{dataset}/x_init_{reduction_rate}_{expID}.pt", map_location="cpu")
        return init_syn_x

    def get_init_syn_eigenvecs(self, n_syn, num_classes):
        n_nodes_per_class = n_syn // num_classes
        n_nodes_last = n_syn % num_classes

        size = [n_nodes_per_class for i in range(num_classes - 1)] + (
            [n_syn - (num_classes - 1) * n_nodes_per_class] if n_nodes_last != 0 else [n_nodes_per_class]
        )
        prob_same_community = 1 / num_classes
        prob_diff_community = prob_same_community / 3

        prob = [
            [prob_diff_community for i in range(num_classes)]
            for i in range(num_classes)
        ]
        for idx in range(num_classes):
            prob[idx][idx] = prob_same_community

        syn_graph = nx.stochastic_block_model(size, prob)
        syn_graph_adj = nx.adjacency_matrix(syn_graph)
        syn_graph_L = normalize_adj(syn_graph_adj)
        syn_graph_L = np.eye(n_syn) - syn_graph_L
        _, eigen_vecs = get_eigh(syn_graph_L, "", False)

        return torch.FloatTensor(eigen_vecs).cuda()

    def generate_labels_syn(self, train_label, reduction_rate):
        from collections import Counter
        
        n = len(train_label)
        counter = Counter(train_label.cpu().numpy())

        num_class_dict = {}

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        y_syn = []
        self.syn_class_indices = {}
       
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * reduction_rate) - sum_
                self.syn_class_indices[c] = [len(y_syn), len(y_syn) + num_class_dict[c]]
                y_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(y_syn), len(y_syn) + num_class_dict[c]]
                y_syn += [c] * num_class_dict[c]

        return y_syn

    def generate_labels_middle(self, train_label, reduction_rate):
        from collections import Counter

        n = len(train_label)
        counter = Counter(train_label.cpu().numpy())

        num_class_dict = {}

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        y_syn = []
        self.syn_class_indices_middle = {}

        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * reduction_rate) - sum_
                self.syn_class_indices_middle[c] = [len(y_syn), len(y_syn) + num_class_dict[c]]
                y_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices_middle[c] = [len(y_syn), len(y_syn) + num_class_dict[c]]
                y_syn += [c] * num_class_dict[c]

        return y_syn
    
    def compute_class_means(self,x_syn, y_syn):
    
        unique_classes = y_syn.unique()  
        class_means = []

        for cls in unique_classes:
        
            mask = (y_syn == cls)
            class_data = x_syn[mask]  
            class_means.append(class_data.mean(dim=0))

        return torch.stack(class_means)