"""Fixed-epoch DLPFC evaluation, adapted from the original DLPFC_test.py.

The model is trained for the number of epochs in the config and clustered
once, after training. Reference labels are never read here; ARI must be
computed separately by aligning predictions.csv to the benchmark metadata.
"""
import argparse
import os
import random
import shutil

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans

from config import Config
from models import Spatial_MGCN
from utils import (ZINB, consistency_loss, normalize_sparse_matrix,
                   regularization_loss, sparse_mx_to_torch_sparse_tensor)


def load_data(path):
    adata = ad.read_h5ad(path)
    x = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    features = torch.FloatTensor(np.asarray(x, dtype=np.float32))
    fadj = adata.obsm['fadj']
    sadj = adata.obsm['sadj']
    nfadj = normalize_sparse_matrix(fadj + sp.eye(fadj.shape[0]))
    nsadj = normalize_sparse_matrix(sadj + sp.eye(sadj.shape[0]))
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)
    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    graph_nei = torch.FloatTensor(np.asarray(adata.obsm['graph_nei'], dtype=np.float32))
    if 'graph_neg' in adata.obsm:
        graph_neg = torch.FloatTensor(np.asarray(adata.obsm['graph_neg'], dtype=np.float32))
    else:
        graph_neg = 1.0 - graph_nei
    return adata, features, nfadj, nsadj, graph_nei, graph_neg


def zinb_loss(pi, disp, features, mean):
    """Average ZINB loss over genes in chunks to fit smaller GPUs."""
    total = 0.0
    for start in range(0, features.shape[1], 1000):
        genes = slice(start, start + 1000)
        width = pi[:, genes].shape[1]
        total = total + width * ZINB(pi[:, genes], theta=disp[:, genes],
                                    ridge_lambda=0).loss(features[:, genes],
                                                         mean[:, genes], mean=True)
    return total / features.shape[1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description='Fixed-epoch DLPFC evaluation.')
    parser.add_argument('--input', required=True, help='Preprocessed Spatial_MGCN.h5ad for one DLPFC slice.')
    parser.add_argument('--output', required=True, help='New directory for predictions and embedding.')
    parser.add_argument('--clusters', required=True, type=int, help='Known number of spatial domains for this slice.')
    parser.add_argument('--config', default=os.path.join(here, 'config', 'DLPFC_fix.ini'))
    args = parser.parse_args()

    config = Config(args.config)
    adata, features, fadj, sadj, graph_nei, graph_neg = load_data(args.input)
    if features.shape[1] != config.fdim:
        raise ValueError('The input gene count must match fdim in the config.')
    if not 2 <= args.clusters < features.shape[0]:
        raise ValueError('clusters must be between 2 and the number of spots minus one.')

    cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if cuda:
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    features = features.to(device)
    fadj = fadj.to(device)
    sadj = sadj.to(device)
    graph_nei = graph_nei.to(device)
    graph_neg = graph_neg.to(device)

    model = Spatial_MGCN(nfeat=features.shape[1], nhid1=config.nhid1,
                         nhid2=config.nhid2, dropout=config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)

    output = None
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        com1, com2, emb, pi, disp, mean = model(features, sadj, fadj)
        gene_loss = zinb_loss(pi, disp, features, mean)
        reg_loss = regularization_loss(emb, graph_nei, graph_neg)
        con_loss = consistency_loss(com1, com2)
        total_loss = config.alpha * gene_loss + config.beta * con_loss + config.gamma * reg_loss
        if epoch == config.epochs - 1:
            # Same convention as the original train(): embedding before this update.
            output = emb.detach().cpu().numpy().copy()
        total_loss.backward()
        optimizer.step()
        print('%s epoch %03d/%03d loss %.4f' % (os.path.basename(args.input),
              epoch + 1, config.epochs, float(total_loss.detach())), flush=True)

    if output is None:
        raise RuntimeError('No embedding was produced; check epochs in the config.')
    labels = KMeans(n_clusters=args.clusters, n_init=10,
                    random_state=config.seed).fit_predict(output)

    os.makedirs(args.output, exist_ok=False)
    pd.DataFrame({'barcode': list(adata.obs_names), 'cluster': labels}).to_csv(
        os.path.join(args.output, 'predictions.csv'), index=False)
    np.save(os.path.join(args.output, 'embedding.npy'), output)
    shutil.copyfile(args.config, os.path.join(args.output, 'config.ini'))
    print('saved', args.output, 'n=%d clusters=%d' % (len(labels), args.clusters), flush=True)


if __name__ == '__main__':
    main()

