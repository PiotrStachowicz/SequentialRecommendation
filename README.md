# ...
...
## Engineering Thesis Context
...
##### Thesis Authors: Jakub Malczak, Piotr Stachowicz 

## DIF-SR
SIGIR 2022 Paper [**"Decoupled Side Information Fusion for Sequential Recommendation"**](https://arxiv.org/abs/2204.11046)

## Overview
We propose DIF-SR to effectively fuse side information for SR via
moving side information from input to the attention layer, motivated
by the observation that early integration of side information and
item id in the input stage limits the representation power of attention
matrices and flexibility of learning gradient. Specifically, we present
a novel decoupled side information fusion attention mechanism,
which allows higher rank attention matrices and adaptive gradient
and thus enhances the learning of item representation. Auxiliary
attribute predictors are also utilized upon the final representation
in a multi-task training scheme to promote the interaction of side
information and item representation.

![avatar](dif.png)

## Preparation

Our code is based on PyTorch 1.8.1 and runnable for both windows and ubuntu server. Required python packages:

> + numpy==1.20.3
> + scipy==1.6.3
> + torch==1.8.1
> + tensorboard==2.7.0

There is also a prepared requirements.txt file. You may install required dependencies via:
```pip install -r requirements.txt```

###### You may experience some errors with torch cuda installation, since requirements.txt was made for nvidia rtx4060

## Usage

Download datasets from [RecSysDatasets](https://github.com/RUCAIBox/RecSysDatasets) or their [Google Drive](https://drive.google.com/drive/folders/1ahiLmzU7cGRPXf5qGMqtAChte2eYp9gI). And put the files in `./dataset/` like the following.

```
$ tree
.
├── Amazon_Beauty
│   ├── Amazon_Beauty.inter
│   └── Amazon_Beauty.item
├── Amazon_Toys_and_Games
│   ├── Amazon_Toys_and_Games.inter
│   └── Amazon_Toys_and_Games.item
├── Amazon_Sports_and_Outdoors
│   ├── Amazon_Sports_and_Outdoors.inter
│   └── Amazon_Sports_and_Outdoors.item
└── yelp
    ├── README.md
    ├── yelp.inter
    ├── yelp.item
    └── yelp.user

```

Run `DIF.sh`.

## Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole).

## Citation
#### Title:
Decoupled Side Information Fusion for Sequential Recommendation

#### Book title:
International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)

#### Year:
2022

##### Authors:
Yueqi Xie (yxieay@connect.ust.hk)
Peilin Zhou (zhoupalin@gmail.com)
Russell Kim (russellkim@upstage.ai)
