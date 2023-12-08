# Spatial-MGCN: a novel multi-view graph convolutional network for identifying spatial domains with attention mechanism

![model](https://github.com/cs-wangbo/Spatial-MGCN/blob/master/Spatial-MGCN/result/Spatial-MGCN.png)


## Requirements 
Python==3.8.13

numpy==1.20.0

pandas==1.4.4

scipy==1.8.1

stlearn==0.4.8

pytorch==1.11.0

torch_geometric==2.1.0

torch_sparse==0.6.15

torch_scatter==2.0.9

matplotlib==3.5.3



## Example

### 1 Raw data 

Raw data should be placed in the folder ***data***.

we put the DLPFC dataset in ***data/DLPFC***. Need to be extracted firstly.

For 10x Spatial Transcripts (ST) datasets, files should be put in the same structure with that provided by 10x website. Taking DLPFC for instance:

> data/DLPFC/151507/ 
  >> spatial/  # The folder where files for spatial information can be found 
  
  >> metadata.tsv # mainly for annotation
  
  >> filtered_feature_bc_matrix.h5 # gene expression data


### 2 Configuration

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



### 3 Data Preprocessing and Graph Construction

Run ***Spatial-MGCN/DLPFC_generate_data.py*** to preprocess the raw DLPFC data:

`python DLPFC_generate_data.py`

Augments:

**--savepath**: the path to save the generated file.

For dealing other ST datasets, please modify the data name. 


### 4 Usage

For training Spatial-MGCN model, run

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

Additionally, the data employed in this study has been uploaded to Zenodo https://zenodo.org/records/10279295

## Citation

Wang et al. Spatial-MGCN: a novel multi-view graph convolutional network for identifying spatial domains with attention mechanism. ***Briefings in Bioinformatics***, 2023, 24(5): bbad262.
