from __future__ import division
from __future__ import print_function

import torch.optim as optim
from utils import *
from models import Spatial_MGCN
import os
import argparse
from config import Config
import pandas as pd


def load_data(dataset):
    print("load data")
    path = "../generate_data/" + dataset + "/Spatial_MGCN.h5ad"
    adata = sc.read_h5ad(path)
    features = torch.FloatTensor(adata.X)
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.LongTensor(adata.obsm['graph_nei'])
    graph_neg = torch.LongTensor(adata.obsm['graph_neg'])
    print("done")
    return features, nsadj, nfadj, graph_nei, graph_neg


def train():
    model.train()
    optimizer.zero_grad()
    com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
    zinb_loss = ZINB(pi, theta=disp, ridge_lambda=0).loss(features, mean, mean=True)
    reg_loss = regularization_loss(emb, graph_nei, graph_neg)
    con_loss = consistency_loss(com1, com2)
    total_loss = config.alpha * zinb_loss + config.beta * con_loss + config.gamma * reg_loss
    emb = pd.DataFrame(emb.cpu().detach().numpy()).fillna(0).values
    mean = pd.DataFrame(mean.cpu().detach().numpy()).fillna(0).values
    total_loss.backward()
    optimizer.step()
    return emb, mean, zinb_loss, reg_loss, con_loss, total_loss


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Mouse_Olfactory']
    for i in range(len(datasets)):
        dataset = datasets[i]
        savepath = './result/' + dataset + '/'
        config_file = './config/' + dataset + '.ini'
        hire_path = '../data/' + dataset + '/spatial/tissue_hires_image.png'
        img = plt.imread(hire_path)

        if not os.path.exists(savepath):
            os.mkdir(savepath)
        print(dataset)

        features, sadj, fadj, graph_nei, graph_neg = load_data(dataset)

        config = Config(config_file)
        cuda = not config.no_cuda and torch.cuda.is_available()
        use_seed = not config.no_seed

        config.n = features.shape[0]
        config.class_num = 7

        if cuda:
            features = features.cuda()
            sadj = sadj.cuda()
            fadj = fadj.cuda()
            graph_nei = graph_nei.cuda()
            graph_neg = graph_neg.cuda()

        config.epochs = config.epochs + 1
        import random

        np.random.seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        if not config.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

        print(dataset, ' ', config.lr, ' ', config.alpha, ' ', config.beta, ' ', config.gamma)
        model = Spatial_MGCN(nfeat=config.fdim,
                            nhid1=config.nhid1,
                            nhid2=config.nhid2,
                            dropout=config.dropout)
        if cuda:
            model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            emb, mean, zinb_loss, reg_loss, con_loss, total_loss = train()
            print(dataset, ' epoch: ', epoch, ' zinb_loss = {:.2f}'.format(zinb_loss),
                  ' reg_loss = {:.2f}'.format(reg_loss), ' con_loss = {:.2f}'.format(con_loss),
                  ' total_loss = {:.2f}'.format(total_loss))
        adata.obsm['emb'] = emb
        adata.obsm['emb'].to_csv(savepath + 'Spatial_MGCN_emb.csv', header=None, index=None)
        adata = mclust_R(adata, num_cluster=config.class_num)
        adata.obs['mclust'].to_csv(savepath + 'Spatial_MGCN_idx.csv', header=None, index=None)

        pl = ['#1f77b4ff', '#ff7f0eff', '#2ca02cff', '#d62728ff', '#9467bdff', '#8c564bff',
              '#e377c2ff']

        plt.axis('off')
        ax = plt.gca()
        ax.imshow(img, extent=[5740, 12410, 9750, 15420])
        sc.pl.embedding(adata, basis="spatial", color="mclust", s=30, show=False,
                        title='Spatial_MGCN', palette=pl, ncols=2, vmin=0, vmax='p99.2')
        plt.savefig(savepath + 'Spatial_MGCN.jpg', bbox_inches='tight', dpi=600)
        plt.show()
        plt.close()
        adata.layers['X'] = adata.X
        adata.layers['mean'] = mean
        adata.write(savepath + 'Spatial_MGCN.h5ad')

        adata.obs['mclust'].replace(1, 'GL', inplace=True)
        adata.obs['mclust'].replace(2, 'GCL', inplace=True)
        adata.obs['mclust'].replace(3, 'IPL', inplace=True)
        adata.obs['mclust'].replace(4, 'ONL', inplace=True)
        adata.obs['mclust'].replace(5, 'EPL', inplace=True)
        adata.obs['mclust'].replace(6, 'RMS', inplace=True)
        adata.obs['mclust'].replace(7, 'MCL', inplace=True)

        n_type = config.class_num
        zeros = np.zeros([adata.n_obs, n_type])
        cell_type = list(adata.obs['mclust'].unique())
        cell_type = [str(s) for s in cell_type]
        cell_type.sort()
        matrix_cell_type = pd.DataFrame(zeros, index=adata.obs_names, columns=cell_type)
        for cell in list(adata.obs_names):
            ctype = adata.obs.loc[cell, 'mclust']
            matrix_cell_type.loc[cell, str(ctype)] = 1

        adata.obs[matrix_cell_type.columns] = matrix_cell_type.astype(str)
        savepath = savepath + 'align/'
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        cell_types = ['EPL', 'GCL', 'GL', 'IPL', 'MCL', 'ONL', 'RMS', ]
        for j in range(len(cell_types)):
            sc.pl.embedding(adata, basis="spatial", color=cell_types[j], s=10, palette=['gray', pl[j]],
                            show=False, vmin=0, vmax='p99.2')
            plt.savefig(savepath + cell_types[j] + '.jpg',
                        bbox_inches='tight', dpi=600)
            plt.show()