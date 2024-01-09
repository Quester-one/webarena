import os
import json
from collections import Counter
from transformers import BertTokenizer, BertModel
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from config_private import http_proxy, https_proxy
import os

os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy


def encode_and_visualize(strings, model_name='bert-base-uncased', tsne_components=2):
    # 加载BERT模型和分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # 编码字符串
    string_embeddings = []

    for string in strings:
        # 分词并编码
        input_ids = tokenizer(string, return_tensors='pt')['input_ids']
        with torch.no_grad():
            outputs = model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            string_embeddings.append(torch.tensor(embeddings))

    # 转换为NumPy数组
    X = torch.stack(string_embeddings).numpy()

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=tsne_components, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # 绘制可视化图
    plt.figure(figsize=(10, 8))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='o', color='b', label='Strings')
    # for i, txt in enumerate(strings):
    #     plt.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.xlabel(f't-SNE Component 1')
    plt.ylabel(f't-SNE Component 2')
    plt.title(f't-SNE Visualization of {model_name} String Embeddings')
    plt.legend()
    plt.show()


def count_and_percentage_list(input_list):
    sets_counter = Counter(map(frozenset, input_list))
    total_sets = len(input_list)
    result = []
    for set_item, count in sets_counter.items():
        percentage = (count / total_sets) * 100
        result.append({
            '集合': set_item,
            '数量': count,
            '比例': percentage
        })
    return result


def count_and_percentage_str(input_list):
    count = {}
    for item in input_list:
        count[item] = count.get(item, 0) + 1
    total_items = len(input_list)
    percentage = {}
    for item, item_count in count.items():
        percentage[item] = (item_count / total_items) * 100
    return {
        '集合': set(input_list),
        '数量': count,
        '比例': percentage
    }


def read_json_files_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    json_files = [file for file in file_list if file[:-5].isdigit() and file.endswith('.json')]
    data_list = []
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {json_file}: {e}")
    return data_list


def extract_values_by_key_set(list_of_dicts, key):
    return [set(d[key]) for d in list_of_dicts if key in d]


def extract_values_by_key_str(list_of_dicts, key):
    return [str(d[key]) for d in list_of_dicts if key in d]


def extract_eval_types(nested_list):
    eval_types_list = [item['eval']['eval_types'] for item in nested_list if
                       'eval' in item and 'eval_types' in item['eval']]
    return eval_types_list


if __name__ == "__main__":
    json_data = read_json_files_in_folder("/data/mentianyi/code/webarena/config_files/")
    sites_list = extract_values_by_key_set(json_data, 'sites')
    sites_dict = count_and_percentage_list(sites_list)
    reset_list = extract_values_by_key_str(json_data, 'require_reset')
    reset_dict = count_and_percentage_str(reset_list)
    eval_type_list = extract_eval_types(json_data)
    eval_type_dict = count_and_percentage_list(eval_type_list)
    intent_list = extract_values_by_key_str(json_data, 'intent')
    intent_dict = count_and_percentage_str(intent_list)
    encode_and_visualize(intent_list)
    print(123)
