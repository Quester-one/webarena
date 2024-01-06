import json
from flask import Flask, render_template_string,send_from_directory
import os
from config_private import VISIALIZE_host,VISIALIZE_port

app = Flask(__name__)

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


def create_mapping_dict(input_dict):
    keys_list = list(input_dict.keys())
    mapping_dict = {}
    reverse_mapping_dict = {}

    for i, key in enumerate(keys_list, start=1):
        mapping_dict[i] = key
        reverse_mapping_dict[key] = i

    return mapping_dict, reverse_mapping_dict

@app.route("/")
def example() -> str:
    from pyvis.network import Network

    # 创建一个网络对象
    net = Network()

    # 添加节点和边缘
    print(node_list)
    for (id,label,image) in node_list:
        net.add_node(id,label=label,shape="image",image=image,size=30)
    for head,tail in edge_list:
        net.add_edge(head,tail)


    # 保存图谱为HTML文件并在浏览器中打开
    net.show('web_docu.html', notebook=False)
    return render_template_string(net.html)

@app.route('/<filename>')
def serve_image(filename):
    images_directory = docu_path
    return send_from_directory(images_directory, filename)

if __name__ == "__main__":
    docu_path = "../web_docu"
    image_names = read_images_from_folder(docu_path)
    docu_dict = read_json_file("../web_docu/my_dict.json")
    id2name_dict, name2id_dict = create_mapping_dict(docu_dict)
    edge_list = []
    for key, value in docu_dict.items():
        for inner_key, inner_value in value['sub_website'].items():
            edge_list.append([name2id_dict[key], name2id_dict[inner_key]])
    node_list=[]
    for key,value in docu_dict.items():
        node_list.append([name2id_dict[key],key+"\n"+value["url"],value["image"]])
    app.run(host=VISIALIZE_host, port=VISIALIZE_port)
    print("end")
