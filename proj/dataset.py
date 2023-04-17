# @Author  : Edlison
# @Date    : 4/12/23 19:12
import numpy as np
import torch
from torch import Tensor
import os
from typing import Optional, Tuple, Dict

AMAZON_DIR = '../data'

AVAILABLE_DATASETS = [
    'Amazon-Video_Games',
    'Amazon-Musical_Instruments',
    'Amazon-Grocery_and_Gourmet_Food',
]


class DynRecDataset(object):
    def __init__(self, name='Amazon-Video_Games'):
        assert name in AVAILABLE_DATASETS, '{} is not available'.format(name)

        # Amazon review data
        if 'Amazon' in name:
            amazon_category = name.split('-')[-1]
            self._load_amazon_dataset(amazon_category)

    def _load_amazon_dataset(self, name):
        processed_dir = os.path.join(AMAZON_DIR, name, 'processed')
        data_dict = torch.load(os.path.join(processed_dir, 'data_dict.pt'))

        # set num_users, num_items (total number)
        self._num_users, self._num_items = data_dict['num_users'], data_dict['num_items']

        # set edge_index_useritem
        useritem = np.stack([data_dict['user'], data_dict['item']])
        self._edge_index_useritem = torch.from_numpy(useritem).to(torch.long)

        # set edge_timestamp
        self._edge_timestamp = torch.from_numpy(data_dict['timestamp']).to(torch.long)
        self._edge_timestamp_ratio = np.linspace(0, 1, len(self._edge_timestamp))

        # set rating
        self._edge_rating = torch.from_numpy(data_dict['rating']).to(torch.float)

        #### set item_attr_dict
        # todo item_attr和item_attr_offset?
        self._item_attr_dict = {}
        self._item_attr_offset_dict = {}
        self.num_item_attrs_dict = {}

        ### item_attr for category
        self._item_attr_dict['category'] = torch.from_numpy(data_dict['category'][:, 1])
        self.num_item_attrs_dict['category'] = data_dict['num_categories']
        unique_item, _item_attr_offset_tmp = np.unique(data_dict['category'][:, 0], return_index=True)
        if unique_item[0] != 0:
            unique_item = np.insert(unique_item, 0, 0)
            _item_attr_offset_tmp = np.insert(_item_attr_offset_tmp, 0, 0)

        _item_attr_offset = [0]
        for i in range(1, len(unique_item)):
            _item_attr_offset.extend([_item_attr_offset_tmp[i]] * (unique_item[i] - unique_item[i - 1]))

        if unique_item[-1] < self._num_items - 1:
            for i in range(self._num_items - unique_item[-1] - 1):
                _item_attr_offset.append(len(data_dict['category'][:, 1]))

        self._item_attr_offset_dict['category'] = torch.from_numpy(np.array(_item_attr_offset))

        ### item_attr for brand
        self._item_attr_dict['brand'] = torch.from_numpy(data_dict['brand'][:, 1])
        self.num_item_attrs_dict['brand'] = data_dict['num_brands']
        unique_item, _item_attr_offset_tmp = np.unique(data_dict['brand'][:, 0], return_index=True)
        if unique_item[0] != 0:
            unique_item = np.insert(unique_item, 0, 0)
            _item_attr_offset_tmp = np.insert(_item_attr_offset_tmp, 0, 0)

        _item_attr_offset = [0]
        for i in range(1, len(unique_item)):
            _item_attr_offset.extend([_item_attr_offset_tmp[i]] * (unique_item[i] - unique_item[i - 1]))

        if unique_item[-1] < self._num_items - 1:
            for i in range(self._num_items - unique_item[-1] - 1):
                _item_attr_offset.append(len(data_dict['brand'][:, 1]))

        self._item_attr_offset_dict['brand'] = torch.from_numpy(np.array(_item_attr_offset))

    def edge_index_useritem(self, time: Optional[float] = None) -> torch.LongTensor:
        if time is None:
            return self._edge_index_useritem
        else:
            return self._edge_index_useritem[:, self._edge_timestamp_ratio <= time]

    def edge_timestamp(self, time: Optional[float] = None) -> torch.LongTensor:
        if time is None:
            return self._edge_timestamp
        else:
            return self._edge_timestamp[self._edge_timestamp_ratio <= time]

    def edge_rating(self, time: Optional[float] = None) -> torch.FloatTensor:
        if time is None:
            return self._edge_rating
        else:
            return self._edge_rating[self._edge_timestamp_ratio <= time]

    def num_users(self, time: Optional[float] = None) -> int:
        return int(self.edge_index_useritem(time)[0].max() + 1)

    def num_items(self, time: Optional[float] = None) -> int:
        return int(self.edge_index_useritem(time)[1].max() + 1)

    def item_attr_pair_dict(self, time: Optional[float] = None) -> Dict[str, Tuple[torch.LongTensor, torch.LongTensor]]:
        '''
            Return a disctionary of pairs of (item_attr, item_attr_offset).
            Consider all kinds of available attributes
            Useful as input to torch.nn.EmbeddingBag

            item_attr: data_dict['brand'][:, 1] or data_dict['category'][:, 1]
            item_attr_offsets:

            return: {'brand': (item_attr, item_attr_offset), 'category': item_attr, item_attr_offset)}

        '''

        num_items = self.num_items(time)
        if time is None or num_items == self._num_items:
            item_attr_pair_dict = {}
            for item_attr_name in self._item_attr_dict.keys():
                item_attr = self._item_attr_dict[item_attr_name]
                item_attr_offset = self._item_attr_offset_dict[item_attr_name]
                item_attr_pair_dict[item_attr_name] = (item_attr, item_attr_offset)
            return item_attr_pair_dict  #
        else:
            item_attr_pair_dict = {}
            for item_attr_name in self._item_attr_dict.keys():
                item_attr_offset = self._item_attr_offset_dict[item_attr_name][:num_items]
                item_attr = self._item_attr_dict[item_attr_name][
                            :self._item_attr_offset_dict[item_attr_name][num_items]]
                item_attr_pair_dict[item_attr_name] = (item_attr, item_attr_offset)
            return item_attr_pair_dict


class Amazon:
    def __init__(self, relation='also_buy'):
        self.x, self.edge_dict = self.load_data()
        self.edge_index = self.edge_dict[relation]
        self.y = self.get_label()
        self.num_node_features = self.x.shape[1]
        self.num_classes = 5
        self.train_ratio = 0.8
        self.train_mask, self.test_mask = self.gen_mask()

    def load_data(self):
        emb_file = '../data/Video_Games/thre100/emb.pt'
        emb = torch.load(emb_file)
        emb = torch.tensor(emb)

        data_file = '../data/Video_Games/thre100/data_dict.pt'
        data_dict = torch.load(data_file)
        also_buy = torch.permute(torch.tensor(data_dict['also_buy']), [1, 0])
        also_view = torch.permute(torch.tensor(data_dict['also_view']), [1, 0])
        edge_dict = {'also_buy': also_buy, 'also_view': also_view}

        return emb, edge_dict

    def get_label(self):
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

        label = torch.zeros(data_dict['num_items'], dtype=torch.long)
        for i in uni_cat4:
            label[i] = 4
        for i in uni_cat3:
            label[i] = 3
        for i in uni_cat2:
            label[i] = 2
        for i in uni_cat1:
            label[i] = 1
        return label

    def gen_mask(self):
        size = self.x.shape[0]
        train_mask = torch.zeros(size, dtype=torch.bool)
        test_mask = torch.zeros(size, dtype=torch.bool)
        train_index = torch.arange(0, self.train_ratio * size, dtype=torch.long)
        test_index = torch.arange(self.train_ratio * size, size, dtype=torch.long)

        train_mask[train_index] = True
        test_mask[test_index] = True

        return train_mask, test_mask


if __name__ == "__main__":
    # dataset = DynRecDataset("Amazon-Video_Games")
    ds = Amazon()
    print(ds.train_mask, len(ds.train_mask))
    print(ds.test_mask, len(ds.test_mask))
