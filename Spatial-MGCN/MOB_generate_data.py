from __future__ import division
from __future__ import print_function

import json

from utils import features_construct_graph, spatial_construct_graph
import torch
import os
import argparse
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
import anndata as ad
from matplotlib.image import imread
from pathlib import Path


def normalize(adata, highly_genes=3000):
    print("start select HVGs")
    # sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=50)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=highly_genes)
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.X / np.sum(adata.X, axis=1).reshape(-1, 1) * 10000
    sc.pp.scale(adata, zero_center=False, max_value=10)
    return adata


def load_ST_file(dataset, highly_genes, k, k1):
    path = "../data/" + dataset + "/"
    data_path = path + "filtered_feature_bc_matrix.h5ad"
    positions_path = path + "spatial/tissue_positions_list.csv"
    savepath = './result/Mouse_Olfactory/Raw/'
    hires_image = path + 'spatial/tissue_hires_image.png'
    scalefactors_json_file = path + 'spatial/scalefactors_json.json'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    annData = sc.read_h5ad(data_path)
    # genenames = ['Mbp', 'Nrgn', 'Pcp4', 'Gabra1', 'Slc6a11', 'Cck', 'Apod']
    # for plot_gene in genenames:
    #     plt.rcParams["figure.figsize"] = (4, 3)
    #     sc.pl.embedding(annData, basis="spatial", color=plot_gene, s=10, layer='X', show=False, vmin=0, vmax='p99.2')
    #     plt.savefig(savepath + plot_gene + '.jpg',
    #                 bbox_inches='tight', dpi=600)
    #     plt.show()
    adata = normalize(annData, highly_genes=highly_genes)

    positions = pd.read_csv(positions_path, sep=',')
    index_labels = np.array(annData.obs.index)
    for i in range(len(index_labels)):
        index_labels[i] = index_labels[i][5:]
    index_labels = index_labels.astype(int)
    position = pd.DataFrame(columns=['y', 'x'])
    for i in range(len(index_labels)):
        position.loc[i] = positions[positions['barcode'] == index_labels[i]].values[0][5:7]
    positions = np.array(position, dtype=float)
    positions[:, [0, 1]] = positions[:, [1, 0]]

    barcodes = np.array(adata.obs.index)
    names = np.array(adata.var.index)
    # X = adata.X
    X = np.nan_to_num(adata.X, nan=0)

    n = len(positions)
    index = range(0, n, 15)
    X = np.delete(X, index, axis=0)
    positions = np.delete(positions, index, axis=0)
    barcodes = np.delete(barcodes, index, axis=0)

    fadj = features_construct_graph(X, k=k)
    sadj, graph_nei, graph_neg = spatial_construct_graph(positions, k=k1)

    adata = ad.AnnData(pd.DataFrame(X, index=barcodes, columns=names))  # , dtype=adata.dtype)
    adata.var_names_make_unique()
    adata.obs['barcodes'] = barcodes
    adata.var['names'] = names

    adata.obsm["spatial"] = positions

    adata.obsm["fadj"] = fadj
    adata.obsm["sadj"] = sadj
    adata.obsm["graph_nei"] = graph_nei.numpy()
    adata.obsm["graph_neg"] = graph_neg.numpy()

    adata.uns["spatial"] = {}
    adata.uns["spatial"][dataset] = {}
    adata.uns["spatial"][dataset]['images'] = {}
    adata.uns["spatial"][dataset]['images']['hires'] = imread(hires_image)
    # adata.uns["spatial"][dataset]['scalefactors'] = json.loads(Path(scalefactors_json_file).read_bytes())
    adata.var_names_make_unique()
    return adata


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    datasets = ['Mouse_Olfactory']
    for i in range(len(datasets)):
        dataset = datasets[i]
        print(dataset)
        if not os.path.exists("../generate_data/"):
            os.mkdir("../generate_data/")
        savepath = "../generate_data/" + dataset + "/"
        config_file = './config/' + dataset + '.ini'
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        config = Config(config_file)
        adata = load_ST_file(dataset, config.fdim, config.k, config.radius)
        print("saving")
        adata.write(savepath + 'Spatial_MGCN.h5ad')
        print("done")
