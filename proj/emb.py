# @Author  : Edlison
# @Date    : 4/13/23 12:40
import torch
import requests as r
import json


def get_emb():
    global data, item
    mapping_dict = '../data/Video_Games/processed/mapping_dict.pt'
    mapping = torch.load(mapping_dict)
    print(mapping['itemname'][:10])
    url = 'https://api.openai.com/v1/embeddings'
    text = mapping['itemname']
    auth = 'Bearer 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'
    data = {"model": "text-embedding-ada-002", "input": text}
    resp = r.post(url, headers={"Content-Type": "application/json", "Authorization": auth}, data=json.dumps(data))
    data = json.loads(resp.text)
    print('request len: ', len(text))
    print('resp len: ', len(data['data']))
    emb = []
    file = '../data/Video_Games/thre100/emb.pt'
    for item in data['data']:
        emb.append(item['embedding'])
    print('emb len: ', len(emb))
    torch.save(emb, file)


if __name__ == '__main__':
    get_emb()
