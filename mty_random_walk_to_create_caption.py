import argparse
import numpy as np
import json
from config_private import WEBARENA_PYTHON_PATH, http_proxy, https_proxy
import os
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
    plt.show()


def get_id_by_name(info, name):
    info_dict = info['observation_metadata']["text"]['obs_nodes_info']
    for key, value in info_dict.items():
        if name in value["text"]:
            return key


def get_name_by_url(graph, url):
    for key, value in graph.items():
        if value["url"] == url:
            return key


def find_element_positions(matrix, target):
    positions = []
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            if element == target:
                positions.append((i, j))
    return positions


class Graph:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.global_url = set()  # 已经看到的url
        self.global_name = set()  # 已经看到的节点
        self.visited_name = set()  # 已经探索到的节点

    def dfs(self, start, obs, info, env):
        go_back_flag=False
        if start not in self.graph:
            raw_element_list = info['observation_metadata']["text"]["obs_nodes_info"]
            text_values = extract_text_values(raw_element_list)
            # 删除复位元素
            text_values = [item for item in text_values if 'link \'Magento Admin Panel\'' not in item]
            img_obs = obs["image"]
            display_image(img_obs)
            if info["page"].url not in self.global_url:
                go_back_flag = True
                self.global_url.add(info["page"].url)
                self.global_name.update(set(text_values))
                self.graph[start]["url"] = info["page"].url
                self.graph[start]["sub_website"] = [[item] for item in text_values]
            else:
                name = get_name_by_url(graph=self.graph, url=info["page"].url)
                clean_text_values_set = set(text_values) - self.global_name
                self.global_name.update(set(text_values))
                # 把列表转成链的形式
                # 说明网页url没有变，所以能找到他的主页名字
                positions = find_element_positions(self.graph[name]["sub_website"], start)
                for position in positions:
                    if len(clean_text_values_set) == 0:
                        self.graph[name]["sub_website"][position[0]].extend(["end"])

        print(start, end=' \n')
        self.visited_name.add(start)
        if start in self.graph:
            for neighbor_list in self.graph[start]['sub_website']:
                for neighbor in neighbor_list:
                    if neighbor not in self.visited_name and neighbor!="end":
                        action_name = "click"
                        element_id = get_id_by_name(info=info, name=neighbor)
                        action = get_standard_action(action_name=action_name, element_id=element_id)
                        obs, _, _, _, info = env.step(action)
                        go_back_flag=self.dfs(neighbor, obs, info, env)
                        if go_back_flag:
                            action_name = "go_back"
                            action = get_standard_action(action_name=action_name, element_id=None)
                            obs, _, _, _, info = env.step(action)
        return go_back_flag


def dfs_method(obs, info, env):
    graph = Graph()
    graph.dfs(start="homepage", obs=obs, info=info, env=env)
    # while True:
    #     action_name = "click"
    #     element_id = "95"
    #     action = get_standard_action(action_name=action_name, element_id=element_id)
    #     if action["action_type"] == ActionTypes.STOP:
    #         break
    #     obs, _, terminated, _, info = env.step(action)
    #     if terminated:
    #         break


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
    obs, info = env.reset(options={"config_file": config_file})
    dfs_method(obs, info, env)
