import json


def read_json(filepath, encoding='utf-8'):
    with open(filepath, 'r', encoding=encoding) as file_obj:
        return json.load(file_obj)


def write_json(filepath, data, encoding='utf-8', ensure_ascii=False, indent=2):
    with open(filepath, 'w', encoding=encoding) as file_obj:
        json.dump(data, file_obj, ensure_ascii=ensure_ascii, indent=indent)


def calculate_F1_score(path):
    js = read_json(path)
    orig = js[-1]['Original Classification']
    pred = js[-1]['Prediction']
    orig_unique = list(set(orig))
    pred_unique = list(set(pred))
    TP_count = {}
    FP_count = {}
    for i in range(len(pred_unique)):
        tp = 0
        fp = 0
        for j in range(len(pred)):
            if orig[j] == pred[j] and pred[j] == orig_unique[i]:
                tp = tp + 1
            elif orig[j] != pred[j] and pred[j] == orig_unique[i]:
                fp = fp + 1
            TP_count[str(i)] = tp
            FP_count[str(i)] = fp
    F1_list = []
    for k in range(len(pred_unique)):
        f1 = TP_count[str(k)] / (TP_count[str(k)] + FP_count[str(k)])
        F1_list.append(f1)
    return sum(F1_list) / len(pred_unique)


if __name__ == '__main__':
    F1_cora_gat = calculate_F1_score('data_exp\\cora_gat_100.json')
    F1_citeseer_gat = calculate_F1_score('data_exp\\citeseer_gat_100.json')
    F1_pubmed_gat = calculate_F1_score('data_exp\\pubmed_gat_100.json')
    # F1_amazon = calculate_F1_score('data_exp\\amazon_gat_100.json')
    js_gat = [{'F1_cora_gat': F1_cora_gat}, {'F1_citeseer_gat': F1_citeseer_gat}, {'F1_pubmed': F1_pubmed_gat}]
    write_json('data_exp\\F1_gat.json', js_gat)

    F1_cora_gcn = calculate_F1_score('data_exp\\cora_gcn_100.json')
    F1_citeseer_gcn = calculate_F1_score('data_exp\\citeseer_gcn_100.json')
    F1_pubmed_gcn = calculate_F1_score('data_exp\\pubmed_gcn_100.json')
    # F1_amazon = calculate_F1_score('data_exp\\amazon_gcn_100.json')
    js_gcn = [{'F1_cora_gcn': F1_cora_gcn}, {'F1_citeseer_gcn': F1_citeseer_gcn}, {'F1_pubmed_gcn': F1_pubmed_gcn}]
    write_json('data_exp\\F1_gcn.json', js_gcn)


