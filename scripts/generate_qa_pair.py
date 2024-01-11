def read_json_file(file_path):
    """
    读取JSON文件并返回其内容
    """
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {file_path}: {e}")
            return None


def filter_dict(docu_dict):
    """
    删除相同网站的页面
    """
    url_set = set()
    filtered_docu_dict = {}
    for key, value in docu_dict.items():
        if value["url"] not in url_set:
            url_set.add(value["url"])
            filtered_docu_dict[key] = value
    return filtered_docu_dict


def generate_qa_pair(filtered_docu_dict,root_image_path,qa_pair_per_page_num,gen_page_num):
    """
    生成训练qa数据
    """
    train_list = []
    count=0
    for index,(key,value) in enumerate(tqdm(filtered_docu_dict.items())):
        if count>gen_page_num:
            break
        count+=1
        image_name=value["image"]
        caption=value["caption"]
        url=value["url"]
        image_path=os.path.join(root_image_path,image_name)
        image = Image.open(image_path)

        try_times=0
        while try_times<20:
            try:
                ocr_prompt = "Please use a table to output the text in the picture"
                ocr_message = [image, ocr_prompt]
                ocr_result = model_v.generate_content(ocr_message).text

                answer_prompt = "Please choose {} short words with the clearest meaning, " \
                                "just output this word,and separated by *** in following content,\n\n".format(qa_pair_per_page_num)
                answer_messages = [{'role': 'user', 'parts': [answer_prompt + ocr_result]}]
                answer_response = model_t.generate_content(answer_messages).text
                answer_list_raw = list(filter(None, answer_response.split('*')))
                answer_list_raw  = [element for element in answer_list_raw  if element.strip() != ""]
                answer_list=[item for item in answer_list_raw  if len(item)>3]

                qa_dict = {}
                print(answer_list)
                for answer in answer_list:
                    question_prompt = "Suppose you are the owner of a store. Please generate a natural language question to use {} as answer based on the picture.".format(
                        answer)
                    print(question_prompt)
                    question_message = [image, question_prompt]
                    question_result = model_v.generate_content(question_message).text
                    qa_dict[question_result] = {"answer":answer,"url":url}

                train_list.append(qa_dict)
                break
            except Exception as e:
                try_times+=1
                # 捕获异常并输出异常信息
                print(f"try_times{try_times},发生异常: {e}")

        save_train_list(train_list=train_list,
                        saved_path="/data/mentianyi/code/webarena/config_files/generated_data/shopping_admin{}".format(index))
    return train_list

def save_train_list(train_list,saved_path):
    if not os.path.exists(saved_path):
        print(f"The directory {saved_path} does not exist. Creating it now...")
        os.makedirs(saved_path)
    json_data = json.dumps(train_list, ensure_ascii=False)
    file_name = "raw.json"
    with open(os.path.join(saved_path,file_name), "w", encoding="utf-8") as json_file:
        json_file.write(json_data)


if __name__ == "__main__":
    import os
    import json
    import google.generativeai as genai
    from config_private import http_proxy, https_proxy, GEMINI_API_KEY
    from PIL import Image
    from tqdm import tqdm

    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    genai.configure(api_key=GEMINI_API_KEY, transport="rest")
    model_v = genai.GenerativeModel('gemini-pro-vision')
    model_t = genai.GenerativeModel('gemini-pro')

    docu_dict = read_json_file("../web_docu/my_dict_caption.json")
    filtered_docu_dict = filter_dict(docu_dict)
    train_list = generate_qa_pair(filtered_docu_dict=filtered_docu_dict,
                                  root_image_path="../web_docu",
                                  qa_pair_per_page_num=2,
                                  gen_page_num=2000)
    save_train_list(train_list=train_list,saved_path="/data/mentianyi/code/webarena/config_files/generated_data/shopping_admin")
    print("end")
