# Spatial-MGCN: a novel multi-view graph convolutional network for identifying spatial domains with attention mechanism

## Introduction
Motivation: Recent advances in spatial transcriptomics (ST) technologies have enabled gene expression profiles while preserving spatial context. Accurately identifying spatial domains, facilitating downstream analysis, require effectively integrating gene expression profiling and spatial information. Recently, an increasing number of computational methods have been developed for spatial domain detection. However, most existing methods fail to adaptively learn the complex relationship between gene expression and spatial information, which results in sub-optimal performance.

Results: To address the above challenges, we propose a novel multi-view graph convolutional network (GCN) with attention mechanism named Spatial-MGCN for identifying spatial domains. In the framework, we first construct two neighbor graphs by leveraging gene expression profiles and spatial information, respectively. Then we design a multi-view GCN encoder to extract unique embeddings from the feature and spatial graphs, respectively, as well as their shared embeddings by combining both graphs. Finally, a zero-inflated negative binomial decoder is adopted to reconstruct the original expression matrix by capturing the global probability distribution of gene expression profiles. Furthermore, we incorporate a spatial regularization constraint into the features learning to preserve spatial neighbor information in an end-to-end manner. The experimental results show that Spatial-MGCN outperforms state-of-the-art methods consistently in several tasks, including identifying spatial domains, identifying tissue structures with varying spatial resolutions, enhancing gene expression patterns, and dissecting cancer tissue heterogeneity.
Taking advantages of two recent technical development, spatial transcriptomics and graph neural network, we thus introduce CCST, Cell Clustering for Spatial Transcriptomics data with graph neural network, an unsupervised cell clustering method based on graph convolutional network to improve ab initio cell clustering and discovering of novel sub cell types based on curated cell category annotation. CCST is a general framework for dealing with various kinds of spatially resolved transcriptomics.


## Requirements 
Python==3.8.13

numpy==1.20.0

pandas==1.4.4

scipy==1.8.1

stlearn==0.4.8

pytorch== 1.11.0

torch_geometric==2.1.0

torch_sparse==0.6.15

torch_scatter==2.0.9

matplotlib==3.5.3



## Example.

### 1 Raw data 

Raw data should be placed in the folder ***data***.

we put the DLPFC dataset in ***data/DLPFC***. Need to be extracted firstly.

For 10x Spatial Transcripts (ST) datasets, files should be put in the same structure with that provided by 10x website. Taking DLPFC for instance:

> data/DLPFC/151507/ 
  >> spatial/  # The folder where files for spatial information can be found 
  
  >> metadata.tsv # mainly for annotation
  
  >> filtered_feature_bc_matrix.h5 # gene expression data


### 2 Data Preprocessing and Graph Construction

Run ***Spatial-MGCN/DLPFC_generate_data.py*** to preprocess the raw DLPFC data:

`python DLPFC_generate_data.py`

Augments:

**--savepath**: the path to save the generated file.

For dealing other ST datasets, please modify the data name. 

---------------------------------------------------------------------------


### 3 Run Spatial-MGCN 

The Spatial-MGCN model is implemented in ***DLPFC_test.py***. We give examples on DLPFC datasets.

The meaning of each argument in ***config.py*** is listed below.

**--epochs**: the number of training epochs.

**--lr**: the learning rate.

**--weight_decay**: the weight decay.

**--k**: the k-value of the k-nearest neighbor graph, which should be within {8...15}.

**--radius**：the spatial location radius.

**--nhid1**: the dimension of the first hidden layer. 

**--nhid2**: the dimension of the second hidden layer. 

**--dropout**: the dropout rate.

**--alpha**: the value of hyperparameter alpha for zinb_loss, which should be within {0.1,1,10}. 

**--beta**: the value of hyperparameter beta for con_loss, which should be within {0.1,1,10}. 

**--gamma**: the value of hyperparameter gamma for reg_loss, which should be within {0.1,1,10}. 

**--no_cuda**: whether to use GPU.

**--no_seed**: whether to take the random parameter.

**--seed**: the random parameter.

**--fdim**: the number of highly variable genes selected.


### 4 Usage

For training your own model, run

'python DLPFC_test.py'

All results are saved in the result folder. We provide our results in the folder ***result*** for taking further analysis. 

(1) The cell clustering labels are saved in ***Spatial_MGCN_idx.csv***, where the first column refers to cell index, and the last column refers to cell cluster label. 

(2) The trained embedding data are saved in ***Spatial_MGCN_emb.csv***.

For Human_Breast_Cancer and Mouse_Olfactory datasets, the running process is the same as above. You just need to run the command:

'python HBC_test.py'

'python MOB_test.py'



## Download all datasets used in Spatial-MGCN:

The datasets used in this paper can be downloaded from the following websites. Specifically, 

(1) The LIBD human dorsolateral prefrontal cortex (DLPFC) dataset http://spatial.libd.org/spatialLIBD 

(2) the processed Stereo-seq dataset from mouse olfactory bulb tissue https://github.com/JinmiaoChenLab/

(3) 10x Visium spatial transcriptomics dataset of human breast cancer https://support.10xgenomics.com/spatial-gene-expression/datasets/1.1.0/V1_Breast_Cancer_Block_A_Section_1

## Framework

![image](https://github.com/cs-wangbo/Spatial-MGCN/tree/master/Spatial-MGCN/result/Spatial-MGCN.png)