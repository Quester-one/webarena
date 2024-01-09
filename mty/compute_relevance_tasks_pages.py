import re
from tqdm import tqdm

def extract_last_abcde(input_str):
    "提取字符串中最后出现的ABCDE其中一个"
    match = re.search(r'[A-E](?=[^A-E]*$)', input_str)
    if match:
        return match.group(0)
    else:
        return None


def top_k_indices(lst, k):
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=True)[:k]
    return indices


def find_index(lst, target):
    for i, value in enumerate(lst):
        if value == target:
            return i
    return -1


def read_json_files_in_directory(directory_path):
    """
    读取一个路径下的所有json文件
    """
    json_data_list = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                json_data_list.append(json_data)

    return json_data_list


def sort_list_of_dicts(list_of_dicts, key='task_id'):
    """
    一个列表里面都是字典，把列表按照字典的key为task_id的value进行排序
    """
    return sorted(list_of_dicts, key=lambda x: x[key])


def read_json_file(file_path):
    """
    读取json文件
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def extract_titles_captions(document):
    titles = []
    captions = []
    for key, value in document.items():
        titles.append(key)
        captions.append(value["caption"])
    return titles, captions


def convert_human_anno_url_2_id(data, human_anno_list):
    url_list = [value["url"] for key, value in data.items()]
    result_list = []
    for index, item in enumerate(human_anno_list):
        if len(item) == 0:
            result_list.append(-1)
        elif len(item) == 1:
            result_list.append(find_index(url_list, item[0]))
    return result_list

def select_best_by_llm(titles_indices,task,captions,model):
    abcde_to_12345_dict={"A":0,"B":1,"C":2,"D":3,"E":4}
    best_id=-1
    selected_captions=[captions[i] for i in titles_indices]
    prompt="For the task:{}, the following are descriptions of five web pages. " \
           "Please choose the most relevant web page from the following pages, " \
           "think about it, and then choose one of ABCDE in the format of '''Options'''\n".format(task)
    for index,option in enumerate(["A","B","C","D","E"]):
        prompt=prompt+"\n******************\n"+option+":\n"+selected_captions[index]
        messages = [{'role': 'user', 'parts': [prompt]}]
        count=0
        while count<20:
            try:
                response = model.generate_content(messages).text
                best_ch=extract_last_abcde(response)
                best_id=titles_indices[abcde_to_12345_dict[best_ch]]
                break
            except:
                count+=1
                pass
    return best_id


if __name__ == "__main__":
    import json
    import os
    from config_private import http_proxy, https_proxy,GEMINI_API_KEY
    from statistics import mean
    from sentence_transformers import SentenceTransformer, util
    import google.generativeai as genai

    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    genai.configure(api_key=GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-pro')

    #######################################################两个分别检索###################################################################
    # # 读取原始任务
    # task_path = "/data/mentianyi/code/webarena/config_files/classified_data_human/shopping_admin/"
    # tasks = read_json_files_in_directory(task_path)
    # sorted_tasks = sort_list_of_dicts(tasks)
    # # 读取先验文档
    # document = read_json_file('/data/mentianyi/code/webarena/web_docu/my_dict_task.json')
    # titles, captions = extract_titles_captions(document)
    # # 读取人工标注,并返回真实网页对应document的元素id
    # human_anno = read_json_file('/data/mentianyi/code/webarena/human_annotation/human_shopping_admin.json')
    # human_anno_urls = [value["main_url"] for key, value in human_anno.items()]
    # human_anno_ids = convert_human_anno_url_2_id(document, human_anno_urls)
    # # 加载检索器
    # model_name = 'all-mpnet-base-v2'  # 选择合适的SBERT模型#paraphrase-MiniLM-L6-v2
    # model = SentenceTransformer(model_name)
    # titles_embeddings = model.encode(titles, convert_to_tensor=True)
    # captions_embeddings = model.encode(captions, convert_to_tensor=True)
    # # 遍历任务,范围左闭右开,topk是检索的文档数量
    # start_id = 0
    # end_id = 20
    # top_k = 130
    # titles_results = []
    # captions_results = []
    # for i in range(start_id, end_id):
    #     task = sorted_tasks[i]["intent"]
    #     query_embedding = model.encode(task, convert_to_tensor=True)
    #     titles_cosine_scores = util.pytorch_cos_sim(query_embedding, titles_embeddings)[0]
    #     captions_cosine_scores = util.pytorch_cos_sim(query_embedding, captions_embeddings)[0]
    #     # 前k个元素，第i个里面的数字表示这个元素在列表里面排第几
    #     titles_indices = top_k_indices(titles_cosine_scores, top_k)
    #     captions_indices = top_k_indices(captions_cosine_scores, top_k)
    #     human_anno_id = human_anno_ids[i]
    #     if human_anno_id != -1:
    #         titles_results.append(titles_indices.index(human_anno_id))
    #         captions_results.append(captions_indices.index(human_anno_id))
    # titles_result = mean(titles_results)
    # captions_result = mean(captions_results)
    # print("end")


    #####################################################先检索再判别############################################################
    # 读取原始任务
    task_path = "/data/mentianyi/code/webarena/config_files/classified_data_human/shopping_admin/"
    tasks = read_json_files_in_directory(task_path)
    sorted_tasks = sort_list_of_dicts(tasks)
    # 读取先验文档
    document = read_json_file('/data/mentianyi/code/webarena/web_docu/my_dict_task.json')
    titles, captions = extract_titles_captions(document)
    # 读取人工标注,并返回真实网页对应document的元素id
    human_anno = read_json_file('/data/mentianyi/code/webarena/human_annotation/human_shopping_admin.json')
    human_anno_urls = [value["main_url"] for key, value in human_anno.items()]
    human_anno_ids = convert_human_anno_url_2_id(document, human_anno_urls)
    # 加载检索器
    model_name = 'all-mpnet-base-v2'  # 选择合适的SBERT模型#paraphrase-MiniLM-L6-v2
    model = SentenceTransformer(model_name)
    titles_embeddings = model.encode(titles, convert_to_tensor=True)
    # 遍历任务,范围左闭右开,topk是检索的文档数量
    start_id = 0
    end_id = 20
    top_k = 130
    titles_results = []
    llm_results=[]
    for i in tqdm(range(start_id, end_id)):
        task = sorted_tasks[i]["intent"]
        query_embedding = model.encode(task, convert_to_tensor=True)
        titles_cosine_scores = util.pytorch_cos_sim(query_embedding, titles_embeddings)[0]
        # 前k个元素，第i个里面的数字表示这个元素在列表里面排第几
        titles_indices = top_k_indices(titles_cosine_scores, top_k)
        best_id=select_best_by_llm(titles_indices[:5],task,captions,llm_model)
        human_anno_id = human_anno_ids[i]
        if human_anno_id != -1:
            titles_results.append(titles_indices.index(human_anno_id))
            if best_id==human_anno_id:
                llm_results.append(1)
            else:
                llm_results.append(0)
    titles_result = mean(titles_results)
    llm_result = mean(llm_results)
    print("end")
