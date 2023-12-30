import os
import json


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


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"文件夹 {folder_path} 创建成功")
    else:
        print(f"文件夹 {folder_path} 已经存在")


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
    json_data = read_json_files_in_folder("/data/mentianyi/code/webarena/config_files/")
    sites_list = extract_values_by_key_set(json_data, 'sites')
    sites_count = count_tuples(sites_list)
    sites_dict = group_dicts_by_key(json_data, "sites")
    create_folder_if_not_exists("/data/mentianyi/code/webarena/config_files/classified_data/")
    create_classified_data(sites_dict, "/data/mentianyi/code/webarena/config_files/classified_data/")
    print("end")
