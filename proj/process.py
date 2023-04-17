import json
import gzip
from tqdm import tqdm
import torch
import os
import numpy as np
from collections import Counter

processed_dir = 'data/processed'


def extract_meta_review(category='Video_Games'):
    '''
        Extracting meta-information about the products

        asin2meta(dict): asin找meta信息
        reviews(list): (user, asin, rating, time), e.g.('A1JGAP0185YJI6', '0700026657', 4.0, '20150727')
    '''

    processed_dir = '../data/{}/processed'.format(category)
    raw_dir = '../data/{}/raw'.format(category)

    path = '{}/meta_{}.json.gz'.format(raw_dir, category)
    g = gzip.open(path, 'r')
    asin2meta = {}

    n_sample = 1500

    for l in tqdm(g):
        line = json.loads(l)
        meta = {}
        meta['asin'] = line['asin']
        meta['brand'] = line['brand']
        meta['category_list'] = line['category']
        meta['main_category'] = line['main_cat']
        meta['also_view'] = line['also_view']
        meta['also_buy'] = line['also_buy']
        meta['title'] = line['title']

        asin2meta[line['asin']] = meta

        n_sample -= 1
        if n_sample == 0:
            break

    os.makedirs(processed_dir, exist_ok=True)

    torch.save(asin2meta, os.path.join(processed_dir, 'asin2meta.pt'))

    path = '{}/{}_5.json.gz'.format(raw_dir, category)

    g = gzip.open(path, 'r')

    review_list = []
    i = 0

    for l in tqdm(g):
        line = json.loads(l)
        rating = line['overall']

        time = line['reviewTime']
        time = time.replace(',', '')
        splitted = time.split(' ')
        mon = splitted[0].zfill(2)
        day = splitted[1][:2].zfill(2)
        year = splitted[2]
        time = '{}{}{}'.format(year, mon, day)

        asin = line['asin']
        user = line['reviewerID']

        review_list.append((user, asin, rating, time))

    torch.save(review_list, os.path.join(processed_dir, 'reviews.pt'))


def create_graph(category='Video_Games'):
    '''
        Mapping everything into index
    '''

    processed_dir = '../data/{}/processed'.format(category)

    asin2meta = torch.load(os.path.join(processed_dir, 'asin2meta.pt'))
    review_list = torch.load(os.path.join(processed_dir, 'reviews.pt'))
    asinset = asin2meta.keys()

    filtered_review_list = []
    for review in review_list:
        # make sure the all the items have meta information
        # 过滤出有meta信息的reviews
        if review[1] in asinset:
            filtered_review_list.append(review)

    timestamp_list = np.array([int(review[3]) for review in filtered_review_list])
    # sort according to time
    # 按照时间重新排列所有review信息，拿到index
    time_sorted_idx = np.argsort(timestamp_list)
    timestamp_list = timestamp_list[time_sorted_idx]

    # 获得所有User和Item对应的列表
    unmapped_user_list_tmp = [filtered_review_list[i][0] for i in time_sorted_idx]
    unmapped_item_list_tmp = [filtered_review_list[i][1] for i in time_sorted_idx]
    rating_list = np.array([review[2] for review in filtered_review_list])
    rating_list = rating_list[time_sorted_idx]

    # 所有User放到集合里 去重
    unique_user_set = set(unmapped_user_list_tmp)
    unique_item_set = set(unmapped_item_list_tmp)

    # mapping used for indexing (tmp)
    # 这里不是一一对应 只是为了拿到对应的index
    unique_user_list_tmp = sorted(list(unique_user_set))
    unique_item_list_tmp = sorted(list(unique_item_set))

    # 每一个User对应一个index，每一个Item对应一个index
    user2idx_tmp = {user: idx for idx, user in enumerate(unique_user_list_tmp)}
    item2idx_tmp = {item: idx for idx, item in enumerate(unique_item_list_tmp)}
    # 对所有review的所有User转换成index，所有Item转换成index
    mapped_user_list_tmp = np.array([user2idx_tmp[unmapped_user] for unmapped_user in unmapped_user_list_tmp])
    mapped_item_list_tmp = np.array([item2idx_tmp[unmapped_item] for unmapped_item in unmapped_item_list_tmp])

    # find the first appearance of user/item
    # 找每个User/Item第一次出现的index
    _, first_appearance_user = np.unique(mapped_user_list_tmp, return_index=True)
    user_idx_sorted_by_time = np.argsort(first_appearance_user)
    user_idx_remapping = np.zeros(len(unique_user_list_tmp), dtype=np.int32)
    user_idx_remapping[user_idx_sorted_by_time] = np.arange(len(unique_user_list_tmp))

    _, first_appearance_item = np.unique(mapped_item_list_tmp, return_index=True)
    item_idx_sorted_by_time = np.argsort(first_appearance_item)
    item_idx_remapping = np.zeros(len(unique_item_list_tmp), dtype=np.int32)
    item_idx_remapping[item_idx_sorted_by_time] = np.arange(len(unique_item_list_tmp))

    # remap everything based on the first appearances
    unique_user_list = [unique_user_list_tmp[i] for i in user_idx_sorted_by_time]  # 所有user
    unique_item_list = [unique_item_list_tmp[i] for i in item_idx_sorted_by_time]  # 所有item
    user2idx = {user: idx for idx, user in enumerate(unique_user_list)}  # dict: user->idx (unique)
    item2idx = {item: idx for idx, item in enumerate(unique_item_list)}  # dict: item->idx (unique)
    mapped_user_list = user_idx_remapping[mapped_user_list_tmp]  # list: 所有review的user_idx
    mapped_item_list = item_idx_remapping[mapped_item_list_tmp]  # list: 所有review的item_idx

    unique_itemname_list = [asin2meta[item]['title'] for item in unique_item_list]

    print('#Users: ', len(user2idx))
    print('#Items: ', len(item2idx))
    print('#Interactions: ', len(mapped_user_list))

    # process also-view and also-buy
    mapped_also_view_mat = []  # 所有item和它的also_view的index [N, 2]
    mapped_also_buy_mat = []

    unmapped_brand_list = []  # only a single brand is assigned per item 存的是meta信息
    unmapped_category_mat = []  # multiple categories may be assigned per item 存的是meta信息

    for item_idx, item in enumerate(unique_item_list):
        meta = asin2meta[item]
        unmapped_also_view_list = meta['also_view']
        unmapped_also_buy_list = meta['also_buy']

        for also_view_item in unmapped_also_view_list:  # 有几个also view当前item就有几个边
            if also_view_item in item2idx:
                mapped_also_view_mat.append([item_idx, item2idx[also_view_item]])

        for also_buy_item in unmapped_also_buy_list:
            if also_buy_item in item2idx:
                mapped_also_buy_mat.append([item_idx, item2idx[also_buy_item]])

        unmapped_brand_list.append(meta['brand'])

        filtered_category_list = list(filter(lambda x: '</span>' not in x, meta['category_list']))
        unmapped_category_mat.append(filtered_category_list)

    mapped_also_view_mat = np.array(mapped_also_view_mat)  # (num_entries, 2)
    mapped_also_buy_mat = np.array(mapped_also_buy_mat)  # (num_entries, 2)

    unmapped_category_mat_concat = []
    for unmapped_category_list in unmapped_category_mat:
        unmapped_category_mat_concat.extend(unmapped_category_list)

    freq_thresh = 100

    # mapping used for indexing
    # 只保留出现大于5次的brand和category
    cnt = Counter(unmapped_brand_list)
    unique_brand_list = []
    for brand, freq in cnt.most_common(10000):
        if freq >= freq_thresh:
            unique_brand_list.append(brand)

    cnt = Counter(unmapped_category_mat_concat)
    unique_category_list = []
    for category, freq in cnt.most_common(10000):
        if freq >= freq_thresh:
            unique_category_list.append(category)

    # 映射brand->index
    brand2idx = {brand: idx for idx, brand in enumerate(unique_brand_list)}
    category2idx = {category: idx for idx, category in enumerate(unique_category_list)}
    print('brand category')
    print('-Brands: ', len(brand2idx))
    print('-Categories: ', len(category2idx))

    # [item_index, brand_index], [N, 2]
    mapped_brand_mat = []
    for item_idx, brand in enumerate(unmapped_brand_list):
        if brand in brand2idx:
            mapped_brand_mat.append([item_idx, brand2idx[brand]])

    # [item_index, category_index], [N, 2]
    mapped_category_mat = []
    for item_idx, category_list in enumerate(unmapped_category_mat):
        for category in category_list:
            if category in category2idx:
                mapped_category_mat.append([item_idx, category2idx[category]])

    mapped_brand_mat = np.array(mapped_brand_mat)
    mapped_category_mat = np.array(mapped_category_mat)

    data_dict = {}
    data_dict['user'] = mapped_user_list  # 所有review对应的user_index
    data_dict['item'] = mapped_item_list  # 所有review对应的item_index
    data_dict['timestamp'] = timestamp_list  # 所有review对应的timestamp
    data_dict['rating'] = rating_list  # 所有review对应的rating
    data_dict['also_buy'] = mapped_also_buy_mat  # first col also_buy second col 每一个item_index和它所有的also_buy 1:N
    data_dict['also_view'] = mapped_also_view_mat  # first col also_view second col 每一个item_index和它所有的also_buy 1:N
    data_dict['brand'] = mapped_brand_mat  # first col item has brand second col 每一个item_index和它的brand_index 1:1
    data_dict['category'] = mapped_category_mat  # first col item has category second col 每一个item_index和它的category_index 1:N
    data_dict['num_users'] = len(unique_user_list)
    data_dict['num_items'] = len(unique_item_list)
    data_dict['num_brands'] = len(unique_brand_list)
    data_dict['num_categories'] = len(unique_category_list)

    mapping_dict = {}
    mapping_dict['user'] = unique_user_list  # user_index
    mapping_dict['item'] = unique_item_list  # item_index
    mapping_dict['itemname'] = unique_itemname_list  # item_title
    mapping_dict['brand'] = unique_brand_list  # brand_name
    mapping_dict['category'] = unique_category_list  # category_name

    torch.save(data_dict, os.path.join(processed_dir, 'data_dict.pt'))
    torch.save(mapping_dict, os.path.join(processed_dir, 'mapping_dict.pt'))


if __name__ == '__main__':
    category_list = [
        'Video_Games'
    ]

    for category in category_list:
        print('Processing {} ...'.format(category))
        extract_meta_review(category)
        create_graph(category)
        print()
