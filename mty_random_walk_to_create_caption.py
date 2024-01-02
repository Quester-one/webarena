import argparse
import numpy as np
import json
from config_private import WEBARENA_PYTHON_PATH, http_proxy, https_proxy
import os

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


def dfs(obs, info, env):
    while True:
        action_name = "click"
        element_id = "95"
        action = get_standard_action(action_name=action_name, element_id=element_id)
        if action["action_type"] == ActionTypes.STOP:
            break
        obs, _, terminated, _, info = env.step(action)
        if terminated:
            break


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
    shopping_admin:环境比较简单，click,scroll,go_back,go_forward似乎就能遍历所有的页面
    
    
    
    
    
    '''
    args = config()
    env = get_env()
    config_file = init_config(args.config_file)
    obs, info = env.reset(options={"config_file": config_file})
    dfs(obs, info, env)
