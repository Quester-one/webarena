import argparse
import numpy as np
import json
from config_private import WEBARENA_PYTHON_PATH, http_proxy, https_proxy
import os
from PIL import Image
import re
import matplotlib.pyplot as plt
from collections import defaultdict

os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy
from browser_env.auto_login import get_site_comb_from_filepath
import tempfile
import subprocess
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")
    parser.add_argument("--render",
                        help="不动. Render the browser. not headless的参数, True表示能看到实时看到浏览器界面，一般会把这个参数设置为False.",
                        default=False)
    parser.add_argument("--slow_mo", type=int, default=0,
                        help="不动. Slow down the browser by the specified amount.减慢操作速度，方便人看清楚，浮点型，数值越大速度越慢")
    parser.add_argument("--observation_type", choices=["accessibility_tree", "html", "image"],
                        default="accessibility_tree", help="不动.环境的返回数据类型，主要是html和accessibility两种类型之间的区分,"
                                                           "后面发现image类型不能作为输入，只能输入前两种文本，图像是附带的，"
                                                           "playwright必须和html匹配，id_accessibility_tree必须和accessibility_tree匹配")
    parser.add_argument("--current_viewport_only", default=True,
                        help="不动.Only use the current viewport for the observation.True是只使用范围内的文本信息，False是不在范围内的直接被截断")
    parser.add_argument("--viewport_width", type=int, default=1280,
                        help="文本和图片会同时被截取，current_viewport_only为True时，剩余的被截断")
    parser.add_argument("--viewport_height", type=int, default=720,
                        help="文本和图片会同时被截取，current_viewport_only为True时，剩余的被截断")
    parser.add_argument("--save_trace_enabled", default=True, help="不动.是否使用追踪记录，True就是使用，会自动保存在cache下的trace的zip里面")
    parser.add_argument("--sleep_after_execution", type=float, default=2.0, help="不动.每次执行后的休眠时间")

    parser.add_argument("--config_file",
                        default="/data/mentianyi/code/webarena/config_files/special_config/random_walk_caption.json",
                        help="随机游走模式的配置")
    parser.add_argument("--save_path", default="web_docu")

    args = parser.parse_args()

    return args


def get_env():
    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
    )
    return env


def create_none_action() -> Action:
    return {
        "action_type": ActionTypes.NONE,
        "coords": np.zeros(2, dtype=np.float32),
        "element_role": 0,
        "element_name": "",
        "text": [],
        "page_number": 0,
        "url": "",
        "nth": 0,
        "pw_code": "",  # str that requires further processing
        "element_id": "",
        "key_comb": "",
        "direction": "",
        "answer": "",
        "raw_prediction": "",
    }


def get_standard_action(action_name, element_id):
    action = create_none_action()
    if "click" in action_name:
        action.update(
            {
                "action_type": ActionTypes.CLICK,
                "element_id": element_id,
                "element_role": 31,
            }
        )
    elif action_name == "scroll_up":
        action.update(
            {
                "action_type": ActionTypes.SCROLL,
                "direction": "up",
            }
        )
    elif action_name == "scroll_down":
        action.update(
            {
                "action_type": ActionTypes.SCROLL,
                "direction": "down",
            }
        )
    elif action_name == "go_back":
        action.update(
            {
                "action_type": ActionTypes.GO_BACK,
            }
        )
    elif action_name == "go_forward":
        action.update(
            {
                "action_type": ActionTypes.GO_FORWARD,
            }
        )
    elif action_name == "stop":
        action.update({"action_type": ActionTypes.STOP, "answer": ""})
    return action


def init_config(config_file):
    with open(config_file) as f:
        _c = json.load(f)
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])
            comb = get_site_comb_from_filepath(cookie_file_name)
            if not os.path.exists(".auth/tmp"):
                os.makedirs(".auth/tmp")
            temp_dir = tempfile.mkdtemp(dir=".auth/tmp")
            subprocess.run(
                [
                    WEBARENA_PYTHON_PATH,
                    "browser_env/auto_login.py",
                    "--auth_folder",
                    temp_dir,
                    "--site_list",
                    *comb,
                ],
            )
            _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
            assert os.path.exists(_c["storage_state"])
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"  # 更新storage_state后，把原来的config_taskid.json又存储了一遍
            with open(config_file, "w") as f:
                json.dump(_c, f)

    return config_file


def extract_text_values(dictionary_list):
    text_values = []
    for element_id, my_dict in dictionary_list.items():
        match = re.search(r'link \'(.+?)\'', my_dict["text"])
        if match:
            text_values.append(match.group(0))
    return text_values


def display_image(img_obs):
    plt.figure(figsize=(13, 13))
    plt.imshow(img_obs)
    plt.axis('off')
    plt.show()  # plt.clf()    plt.cla()


def get_id_by_name(info, name):
    info_dict = info['observation_metadata']["text"]['obs_nodes_info']
    for key, value in info_dict.items():
        if name in value["text"]:
            return key


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"路径 {path} 不存在，已创建成功。")
    else:
        print(f"路径 {path} 已存在。")


class Graph:
    def __init__(self, args):
        self.args = args
        self.graph = defaultdict(dict)
        self.global_name = set()  # 已经看到的节点
        self.visited_name = set()  # 已经探索到的节点
        self.image_id = 0

    def direct(self, start, obs, info, env):
        print(start, end=' \n')
        self.visited_name.add(start)
        # 如果满足条件，说明是第一次遇到该元素
        if start not in self.graph:
            current_element_name = extract_text_values(info['observation_metadata']["text"]["obs_nodes_info"])
            # 删除复位元素
            current_element_name = [item for item in current_element_name if 'link \'Magento Admin Panel\'' not in item]
            img_obs = obs["image"]
            image = Image.fromarray(img_obs)
            image.save(os.path.join(args.save_path, '{}.png'.format(self.image_id)))
            self.graph[start]["image"] = '{}.png'.format(self.image_id)
            self.image_id = self.image_id + 1
            # display_image(img_obs)
            filtered_element_name = sorted(list(set(current_element_name) - self.global_name))
            self.global_name.update(set(current_element_name))
            self.graph[start]["url"] = info["page"].url
            # self.graph[start]["image"]=img_obs
            filtered_element_name = {key: False for key in filtered_element_name}
            # filtered_element_name={"link '\\ue600 admin'":False}
            if "link 'Sign Out'" in filtered_element_name:
                filtered_element_name.pop("link 'Sign Out'")
            if "link 'Remove'" in filtered_element_name:
                filtered_element_name.pop("link 'Remove'")
            self.graph[start]["sub_website"] = filtered_element_name
            self.graph[start]["all_children_explored"] = False
            print(123)
        else:
            filtered_element_name = self.graph[start]["sub_website"]
            print(456)
        # 到达叶子节点
        if len(filtered_element_name) == 0:
            action_name = "click"
            element_id = get_id_by_name(info=info, name='link \'Magento Admin Panel\'')
            action = get_standard_action(action_name=action_name, element_id=element_id)
            obs, _, _, _, info = env.step(action)
            self.graph[start]["all_children_explored"] = True
        # 到达非叶子节点
        else:
            for child in filtered_element_name:
                if not self.graph[start]["sub_website"][child]:
                    action_name = "click"
                    element_id = get_id_by_name(info=info, name=child)
                    action = get_standard_action(action_name=action_name, element_id=element_id)
                    obs, _, _, _, info = env.step(action)
                    self.direct(child, obs, info, env)
                    if self.graph[child]['all_children_explored']:
                        self.graph[start]["sub_website"][child] = True
                    if all(value for value in self.graph[start]["sub_website"].values()):
                        self.graph[start]['all_children_explored'] = True
                    return None


def direct_method(args, obs, info, env):
    graph = Graph(args)
    while True:
        graph.direct(start="homepage", obs=obs, info=info, env=env)
        if graph.graph["homepage"]["all_children_explored"] == True:
            break
    my_dict = dict(graph.graph)
    with open(os.path.join(args.save_path, "my_dict.json"), 'w') as json_file:
        json.dump(my_dict, json_file, indent=2)


if __name__ == "__main__":
    '''
    动作空间说明
    1.页面操作动作
    click [id]:点击元素id
    type [id] [content] [press_enter_after=0|1]:在元素id中敲字，默认press_enter_after参数是1，指的输入字符后回车
    hover [id]:在元素id上悬停
    press [key_comb]:按下组合键
    scroll [direction=down|up]:向上或向下滚动页面
    2.选项卡管理操作
    new_tab:打开一个新的浏览器选项卡
    tab_focus [tab_index]:切换到tab_index的浏览器选项卡
    close_tab:关闭当前活动的选项卡
    3.url导航操作
    goto [url]:导航到特定的URL
    go_back:导航到之前查看的页面
    go_forward:在go_back的基础上再回退回去
    4.结束
    stop [answer]:问答问题输出的最终答案
    N/A:表示该任务无法完成
    
    输出格式说明
    CoT+```答案```
    例如CoT+```click [1234]```
        
    环境：
    shopping_admin:环境比较简单，click,scroll,go_back,go_forward似乎就能遍历所有的页面.操作元素认为跳转的页面有link  
    '''
    args = config()
    env = get_env()
    config_file = init_config(args.config_file)
    create_directory(args.save_path)
    obs, info = env.reset(options={"config_file": config_file})
    direct_method(args, obs, info, env)
