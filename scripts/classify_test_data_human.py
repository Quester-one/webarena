import os
import re
import json
from collections import Counter


def get_all_files_in_folder(folder_path):
    files = []
    # 遍历文件夹中的所有文件和子文件夹
    for root, dirs, filenames in os.walk(folder_path):
        for filename in filenames:
            # 获取文件的完整路径
            file_path = os.path.join(root, filename)
            files.append(file_path)

    return files


def extract_numbers_from_paths(paths):
    numbers = []

    for path in paths:
        match = re.search(r'(\d+)', path)
        if match:
            number = match.group(1)
            numbers.append(number)
        else:
            numbers.append(None)  # 或者你可以选择添加其他标记来表示未找到数字的情况

    return numbers


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


def count_tuples(lst):
    tuple_count = {}
    for tpl in lst:
        key = str(tpl)
        tuple_count[key] = tuple_count.get(key, 0) + 1
    return tuple_count


def group_dicts_by_key(input_list, key):
    result_dict = {}
    for dictionary in input_list:
        data_domian = dictionary.get(key)
        data_domian.sort()
        data_domian = ','.join(map(str, data_domian))
        key_value = str(data_domian)
        if key_value not in result_dict:
            result_dict[key_value] = []
        result_dict[key_value].append(dictionary)
    return result_dict


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 创建成功")
    else:
        print(f"文件夹 {folder_path} 已经存在")


def create_classified_data(sites_dict, root_path):
    for key, value in sites_dict.items():
        dict_name = "+".join(key.split(','))
        path = os.path.join(root_path, dict_name)
        create_folder_if_not_exists(path)
        for index, item in enumerate(value):
            item["task_id"] = index
            with open(os.path.join(path, "{}.json".format(index)), 'w') as json_file:
                json.dump(item, json_file, indent=4)


if __name__ == "__main__":
    human_id_list = get_all_files_in_folder("/data/mentianyi/code/webarena/webarena_human/trajectories/")
    human_id_list = sorted(map(int, extract_numbers_from_paths(human_id_list)))
    all_file_list = read_json_files_in_folder("/data/mentianyi/code/webarena/config_files/")
    all_file_list = sorted(all_file_list, key=lambda x: x['task_id'])
    human_file_list = [all_file_list[i] for i in human_id_list]

    sites_list = extract_values_by_key_set(human_file_list, 'sites')
    sites_count = count_tuples(sites_list)
    sites_dict = group_dicts_by_key(human_file_list, "sites")
    # create_folder_if_not_exists("../config_files/classified_data_human/")
    # create_classified_data(sites_dict, "../config_files/classified_data_human/")
    print("end")
