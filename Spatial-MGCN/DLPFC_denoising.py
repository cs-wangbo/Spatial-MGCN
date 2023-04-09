import numpy as np
import scanpy as sc
from matplotlib import pyplot as plt
import os

adata = sc.read_h5ad('./result/DLPFC/151507/Spatial_MGCN.h5ad')
savepath = './result/DLPFC/151507/Denoise/'

if not os.path.exists(savepath):
    os.mkdir(savepath)

marker_genes = ['ATP2B4', 'FKBP1A', 'CRYM', 'NEFH', 'RXFP1', 'B3GALT2']#, 'NTNG2'

names=np.array(adata.var_names)

plt.rcParams["figure.figsize"] = (3, 3)
for gene in marker_genes:
    if np.isin(gene, np.array(adata.var_names)):
        sc.pl.spatial(adata, img_key="hires", color=gene, show=False, title=gene, layer='X', vmax='p99')
        plt.savefig(savepath + gene + '_raw.jpg', bbox_inches='tight', dpi=600)
        plt.show()

        sc.pl.spatial(adata, img_key="hires", color=gene, show=False, title=gene, layer='mean',
                      vmax='p99')
        plt.savefig(savepath + gene + '_mean.jpg', bbox_inches='tight', dpi=600)
        plt.show()


sc.pl.stacked_violin(adata, marker_genes, title='Raw', groupby='ground_truth', swap_axes=True,
                     figsize=[6, 3], show=False)
plt.savefig(savepath + 'stacked_violin_Raw.jpg', bbox_inches='tight', dpi=600)
plt.show()

sc.pl.stacked_violin(adata, marker_genes, layer='mean', title='Spatial-MGCN', groupby='ground_truth', swap_axes=True,
                     figsize=[6, 3], show=False)
plt.savefig(savepath + 'stacked_violin_Spatial_MGCN.jpg', bbox_inches='tight', dpi=600)
plt.show()