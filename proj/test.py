# @Author  : Edlison
# @Date    : 4/12/23 17:54
import torch
import numpy as np
import os
from dataset import DynRecDataset


def show():
    global data
    a2m = '../data/Video_Games/processed/asin2meta.pt'
    reviews = '../data/Video_Games/processed/reviews.pt'
    data_dict = '../data/Video_Games/processed/data_dict.pt'
    mapping_dict = '../data/Video_Games/processed/mapping_dict.pt'
    data = torch.load(data_dict)
    mapping = torch.load(mapping_dict)
    n = 10
    print('num brands: ', data['num_brands'], 'num categories: ', data['num_categories'])
    print(data['also_buy'][20:100])
    print(data['also_buy'].shape)
    print(data['also_view'].shape)
    nei_idx = data['also_buy'][:n][:, 1]
    cat_idx = data['category'][nei_idx, 1]
    cat0 = (data['category'][:, 1] == 0).sum()
    cat1 = (data['category'][:, 1] == 1).sum()
    cat2 = (data['category'][:, 1] == 2).sum()
    cat3 = (data['category'][:, 1] == 3).sum()
    cat4 = (data['category'][:, 1] == 4).sum()
    print('cat0 num: ', cat0, cat1, cat2, cat3, cat4)
    print('max', np.max(nei_idx))
    print(data['category'][0, 1])
    print('cat idx: ', cat_idx)
    print(mapping['brand'])
    print(mapping['category'])
    print(len(mapping['item']))
    print(len(mapping['user']))
    print(len(mapping['itemname']))
    print(len(mapping['category']))
    print(len(mapping['brand']))
    print([mapping['itemname'][i] for i in nei_idx])
    print(data['num_brands'], data['num_categories'])


def load_data():
    emb_file = '../data/Video_Games/thre100/emb.pt'
    emb = torch.load(emb_file)
    emb = torch.tensor(emb)

    data_file = '../data/Video_Games/thre100/data_dict.pt'
    data_dict = torch.load(data_file)
    also_buy = torch.permute(torch.tensor(data_dict['also_buy']), [1, 0])
    also_view = torch.permute(torch.tensor(data_dict['also_view']), [1, 0])
    edge_dict = {'also_buy': also_buy, 'also_view': also_view}

    return emb, edge_dict


def get_label():
    data_file = '../data/Video_Games/thre100/data_dict.pt'
    data_dict = torch.load(data_file)

    category = data_dict['category']
    item_idx = category[:, 0]
    cat4_idx = np.argwhere(category[:, 1] == 4)
    cat3_idx = np.argwhere(category[:, 1] == 3)
    cat2_idx = np.argwhere(category[:, 1] == 2)
    cat1_idx = np.argwhere(category[:, 1] == 1)
    cat0_idx = np.argwhere(category[:, 1] == 0)
    cat4_item = category[cat4_idx, 0]
    cat3_item = category[cat3_idx, 0]
    cat2_item = category[cat2_idx, 0]
    cat1_item = category[cat1_idx, 0]
    cat0_item = category[cat0_idx, 0]
    uni_cat4 = cat4_item
    uni_cat3 = [x for x in cat3_item if x not in cat4_item]
    uni_cat2 = [x for x in cat2_item if x not in cat4_item and x not in cat3_item]
    uni_cat1 = [x for x in cat1_item if x not in cat4_item and x not in cat3_item and x not in cat2_item]
    uni_cat0 = [x for x in cat0_item if
                x not in cat4_item and x not in cat3_item and x not in cat2_item and x not in cat1_item]

    label = torch.zeros(data_dict['num_items'])
    for i in uni_cat4:
        label[i] = 4
    for i in uni_cat3:
        label[i] = 3
    for i in uni_cat2:
        label[i] = 2
    for i in uni_cat1:
        label[i] = 1
    return label


if __name__ == '__main__':
    x, edge_dict = load_data()
    label = get_label()

    print(x.shape, label.shape)
    print(edge_dict['also_buy'].shape, edge_dict['also_view'].shape)
    torch.manual_seed(0)
    randidx = torch.randperm(4210)
    print(randidx[0])
    rand_edge = edge_dict['also_buy'][:, randidx]
    print(rand_edge[:, :10])
    print(edge_dict['also_buy'][:, randidx[0]])