import json
import train as tr


def write_json(filepath, data, encoding='utf-8', ensure_ascii=False, indent=2):
    with open(filepath, 'w', encoding=encoding) as file_obj:
        json.dump(data, file_obj, ensure_ascii=ensure_ascii, indent=indent)


if __name__ == '__main__':
    """
        dataset: {'amazon', 'cora', 'CiteSeer', 'PubMed'}
        model: {'GCN', 'GAT'}
    """

    # 100 iterations
    cora_gat_100 = tr.train_exp(dataset_name='cora', model_name='GAT', iterations=100)
    citeseer_gat_100 = tr.train_exp(dataset_name='CiteSeer', model_name='GAT', iterations=100)
    pubmed_gat_100 = tr.train_exp(dataset_name='PubMed', model_name='GAT', iterations=100)
    amazon_gat_100 = tr.train_exp(dataset_name='amazon', model_name='GAT', iterations=100)

    write_json('data_exp\\cora_gat_100.json', cora_gat_100)
    write_json('data_exp\\citeseer_gat_100.json', citeseer_gat_100)
    write_json('data_exp\\pubmed_gat_100.json', pubmed_gat_100)
    write_json('data_exp\\amazon_gat_100.json', amazon_gat_100)
