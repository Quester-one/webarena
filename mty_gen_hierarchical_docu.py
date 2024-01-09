import json
import os
from tqdm import tqdm


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # 读取文件内容
            json_data = file.read()

            # 解析JSON
            data = json.loads(json_data)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file {file_path}: {e}")
        return None


def read_images_from_folder(folder_path):
    image_names = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # 可以根据需要添加其他图片格式

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_names.append(filename)

    return image_names


def gen_function_summary_each_page(docu_path, docu_dict,output):
    import PIL.Image
    import os
    import google.generativeai as genai
    from config_private import http_proxy, https_proxy, GEMINI_API_KEY

    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel('gemini-pro-vision')
    for key, value in tqdm(docu_dict.items()):
        while True:
            try:
                image = PIL.Image.open(os.path.join(docu_path, value["image"]))
                message = [image, "Could you please summarize the main functions of this page"]
                docu_dict[key]["caption"] = model.generate_content(message).text
                break
            except:
                pass
    with open(os.path.join(docu_path, output), 'w') as json_file:
        json.dump(docu_dict, json_file)
    print(123)


if __name__ == "__main__":
    docu_path = "web_docu"
    image_names = read_images_from_folder(docu_path)
    docu_dict = read_json_file("web_docu/my_dict.json")
    gen_function_summary_each_page(docu_path, docu_dict,output="my_dict_task.json")
    print("end")
