import json
import train as tr
import calculate_F1_score as calf1


def write_json(filepath, data, encoding='utf-8', ensure_ascii=False, indent=2):
    with open(filepath, 'w', encoding=encoding) as file_obj:
        json.dump(data, file_obj, ensure_ascii=ensure_ascii, indent=indent)


if __name__ == '__main__':
    """
        dataset: {'amazon', 'cora', 'CiteSeer', 'PubMed'}
        model: {'GCN', 'GAT'}
    """
    # 100 iterations
    # cora_gat_100 = tr.train_exp_return_data_pred(dataset_name='cora', model_name='GAT', iterations=100)
    # citeseer_gat_100 = tr.train_exp_return_data_pred(dataset_name='CiteSeer', model_name='GAT', iterations=100)
    # pubmed_gat_100 = tr.train_exp_return_data_pred(dataset_name='PubMed', model_name='GAT', iterations=100)
    # amazon_gat_100 = tr.train_exp_return_data_pred(dataset_name='amazon', model_name='GAT', iterations=100)

    # write_json('data_exp\\cora_gat_100.json', cora_gat_100)
    # write_json('data_exp\\citeseer_gat_100.json', citeseer_gat_100)
    # write_json('data_exp\\pubmed_gat_100.json', pubmed_gat_100)
    # write_json('data_exp\\amazon_gat_100.json', amazon_gat_100)

    # cora_gcn_100 = tr.train_exp_return_data_pred(dataset_name='cora', model_name='GCN', iterations=100)
    # citeseer_gcn_100 = tr.train_exp_return_data_pred(dataset_name='CiteSeer', model_name='GCN', iterations=100)
    # pubmed_gcn_100 = tr.train_exp_return_data_pred(dataset_name='PubMed', model_name='GCN', iterations=100)
    #
    # write_json('data_exp\\cora_gcn_100.json', cora_gcn_100)
    # write_json('data_exp\\citeseer_gcn_100.json', citeseer_gcn_100)
    # write_json('data_exp\\pubmed_gcn_100.json', pubmed_gcn_100)

    # change GAT model
    # a = tr.train_exp_amazon('GAT-heads', 4, 100, 0.005, 5e-4)
    # b = tr.train_exp_amazon('GAT-heads', 8, 100, 0.005, 5e-4)
    # c = tr.train_exp_amazon('GAT-heads', 16, 100, 0.005, 5e-4)
    # d = tr.train_exp_amazon('GAT-heads', 32, 100, 0.005, 5e-4)
    # e = tr.train_exp_amazon('GAT-heads', 64, 100, 0.005, 5e-4)
    # f = tr.train_exp_amazon('GAT-heads', 128, 100, 0.005, 5e-4)
    # write_json('data_exp\\heads_analysis\\amazon_gat_100_4.json', a)
    # write_json('data_exp\\heads_analysis\\amazon_gat_100_8.json', b)
    # write_json('data_exp\\heads_analysis\\amazon_gat_100_16.json', c)
    # write_json('data_exp\\heads_analysis\\amazon_gat_100_32.json', d)
    # write_json('data_exp\\heads_analysis\\amazon_gat_100_64.json', e)
    # write_json('data_exp\\heads_analysis\\amazon_gat_100_128.json', f)

    # amazon_gcn = tr.train_exp_return_data_pred('amazon', 'GCN')
    amazon_gat = tr.train_exp_amazon('GAT-heads', 8, 100, 0.005, 5e-4)
    # amazon_gsac = tr.train_my_exp(iterations=100, lr=0.005, reg=5e-4)
    # write_json('data_exp\\amazon_gcn_100_view.json', amazon_gcn)
    write_json('data_exp\\also_view\\amazon_gat_100_view_5.json', amazon_gat)
    # write_json('data_exp\\amazon_gsac_100_view.json', amazon_gsac)

    # pubmed_gsac = tr.train_my_exp(iterations=100, lr=0.005, reg=5e-4)
    # write_json('data_exp\\pubmed_gsac_100.json', pubmed_gsac)


