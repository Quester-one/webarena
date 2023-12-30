import argparse
import glob
import json
import logging
from config_private import SHOPPING, SHOPPING_ADMIN, REDDIT, GITLAB, MAP, WIKIPEDIA, HOMEPAGE, http_proxy, \
    https_proxy, OPENAI_API_KEY, WEBARENA_PYTHON_PATH, GEMINI_API_KEY
import os

os.environ["SHOPPING"] = SHOPPING
os.environ["SHOPPING_ADMIN"] = SHOPPING_ADMIN
os.environ["REDDIT"] = REDDIT
os.environ["GITLAB"] = GITLAB
os.environ["MAP"] = MAP
os.environ["WIKIPEDIA"] = WIKIPEDIA
os.environ["HOMEPAGE"] = HOMEPAGE
os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
import random
import subprocess
import tempfile
import time
from pathlib import Path
import openai

from agent import (
    Agent,
    PromptAgent,
    TeacherForcingAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
)
from evaluation_harness import evaluator_router

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def config() -> argparse.Namespace:
    '''
    配置参数
    '''
    # general config
    parser = argparse.ArgumentParser(description="Run end-to-end evaluation on the benchmark")
    parser.add_argument("--render",
                        help="不动. Render the browser. not headless的参数, True表示能看到实时看到浏览器界面，一般会把这个参数设置为False.",
                        default=False)
    parser.add_argument("--render_screenshot", help="不动.Render the browser.html路径记录里面，是否保存屏幕截图，True是保存.", default=True)
    parser.add_argument("--slow_mo", type=int, default=0,
                        help="不动. Slow down the browser by the specified amount.减慢操作速度，方便人看清楚，浮点型，数值越大速度越慢")
    parser.add_argument("--action_set_tag", default="id_accessibility_tree",
                        help="不动.Action type.输出动作的形式，id_accessibility_tree是简化版，playwright是html版本",
                        choices=["id_accessibility_tree", "playwright"])
    parser.add_argument("--observation_type", choices=["accessibility_tree", "html", "image"],
                        default="accessibility_tree", help="不动.环境的返回数据类型，主要是html和accessibility两种类型之间的区分,"
                                                           "后面发现image类型不能作为输入，只能输入前两种文本，图像是附带的，"
                                                           "playwright必须和html匹配，id_accessibility_tree必须和accessibility_tree匹配")
    parser.add_argument("--current_viewport_only", default=True,
                        help="不动.Only use the current viewport for the observation.True是只使用范围内的文本信息，False是不在范围内的直接被截断")
    parser.add_argument("--save_trace_enabled", default=True, help="不动.是否使用追踪记录，True就是使用，会自动保存在cache下的trace的zip里面")
    parser.add_argument("--sleep_after_execution", type=float, default=2.0, help="不动.每次执行后的休眠时间")
    parser.add_argument("--max_steps", type=int, default=30, help="不动.最大实验次数.")
    parser.add_argument("--viewport_width", type=int, default=1280,
                        help="文本和图片会同时被截取，current_viewport_only为True时，剩余的被截断")
    parser.add_argument("--viewport_height", type=int, default=720,
                        help="文本和图片会同时被截取，current_viewport_only为True时，剩余的被截断")
    parser.add_argument("--specific_dataset", default="classified_data/shopping_admin", help="指定选用的数据集子集",
                        choices=["classified_data/gitlab", "classified_data/map", "classified_data/reddit",
                                 "classified_data/shopping", "classified_data/shopping_admin",
                                 "classified_data/gitlab+reddit",
                                 "classified_data/gitlab+wikipedia", "classified_data/map+shopping_admin",
                                 "classified_data/map+wikipedia", "classified_data/reddit+shopping"])

    # agent config

    parser.add_argument("--agent_type", type=str, default="prompt", choices=["teacher_forcing", "prompt"],
                        help="不动.推断时候，teacher_forcing是历史信息使用真值，promot是使用误差累积的方式，本实验全部使用prompt")
    parser.add_argument("--parsing_failure_th", type=int, default=3,
                        help="不动.When concesecutive parsing failure exceeds this threshold, the agent will stop，连续三次解析错误就终止")
    parser.add_argument("--repeating_action_failure_th", type=int, default=3,
                        help="不动.When concesecutive repeating action exceeds this threshold, the agent will stop，连续输出三次相同的操作就终止")
    parser.add_argument("--instruction_path", type=str, default="agent/prompts/jsons/p_cot_id_actree_2s.json",
                        help="prompt模板的路径，总共有5种")

    # lm config
    parser.add_argument("--mode", type=str, default="chat", choices=["chat", "completion"],
                        help="不动.chat是聊天型，completion是补全型，现在用的几个模型都是chat型")
    parser.add_argument("--temperature", type=float, default=1.0, help="没看懂")
    parser.add_argument("--top_p", type=float, default=0.9, help="没看懂")
    parser.add_argument("--context_length", type=int, default=0, help="没看懂")
    parser.add_argument("--max_tokens", type=int, default=384, help="没看懂")
    parser.add_argument("--stop_token", type=str, default=None, help="没看懂")
    parser.add_argument("--max_retry", type=int, help="max retry times to perform generations when parsing fails，没看懂",
                        default=1, )
    parser.add_argument("--max_obs_length", type=int, default=1920,
                        help="when not zero, will truncate the observation to this length before feeding to the model", )
    parser.add_argument("--model_endpoint", help="huggingface model endpoint", type=str, default="", )
    parser.add_argument("--provider", type=str, default="google", choices=["openai", "huggingface", "google"],
                        help="模型的发布机构，用于选择配置参数的字典")
    parser.add_argument("--model", type=str, default='gemini-pro',
                        choices=["gpt-3.5-turbo-0613", 'gemini-pro-vision', 'gemini-pro'], help="具体使用的模型")
    parser.add_argument("--imageassist", type=str, default=False,
                        help="True是使用图像信息辅助，使用gemini-pro-vision时为True，其他为False")

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=50)

    # logging related
    parser.add_argument("--result_dir", type=str, default=None, help="不动.None自动产生时间戳，始终设置为None即可")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if (
            args.action_set_tag == "id_accessibility_tree"
            and args.observation_type != "accessibility_tree"
    ):
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def prepare(args: argparse.Namespace) -> None:
    '''
    通过to_json.run()把agent/prompts/raw里面的py，转化成agent/prompts/json里面的json，里面是5种prompt模板，每次都会覆盖刷新
    result_dir永远设置为None，所以在cache建立结果文件夹，有时间戳
        然后建立log文件，log_files.txt和traces
        建立config文件，把所有参数存储在config.json
    '''
    from agent.prompts import to_json
    to_json.run()

    result_dir = args.result_dir
    if not result_dir:
        result_dir = (f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}")
    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")
    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")

    config_file = Path(args.result_dir) / "config.json"
    if not config_file.exists():
        with open(config_file, "w") as f:
            json.dump(vars(args), f, indent=4)
            logger.info(f"Dump config to {config_file}")


def get_test_file_list():
    '''
    返回路径下还没有测试完的样例
    '''
    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        test_file_list.append(os.path.join("config_files", args.specific_dataset, "{}.json".format(i)))
    if "debug" not in args.result_dir:
        test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    if len(test_file_list) == 0:
        logger.info("No task left to run")
    return test_file_list


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [
        os.path.basename(f).split(".")[0].split("_")[1] for f in result_files
    ]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def test(
        args: argparse.Namespace,
        agent: Agent | PromptAgent | TeacherForcingAgent,
        config_file_list: list[str],
) -> None:
    scores = []
    max_steps = args.max_steps
    early_stop_thresholds = {"parsing_failure": args.parsing_failure_th,
                             "repeating_action": args.repeating_action_failure_th}

    # 初始化环境的各种参数
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

    for config_file in config_file_list:
        '''
        第一层：交互发生错误会终止报错
        第二层：遍历所有任务
        第三层：对于一个任务，一直交互，直到自己终止或者重复输出相同的步数或者达到最大交互步数
        '''
        try:
            intent, task_id, config_file, render_helper = init_config(config_file)
            agent.reset(config_file)  # 该步骤实际什么都没有执行
            trajectory: Trajectory = []
            # obs是文本和图像观测,info是观测到的文本信息的编号，坐标，文本
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            while True:
                # 当超过最大交互次数,parsing_failure是最后三次行动都失败,repeating_action是最后三次相同
                # 如果触发以上条件，early_stop_flag为True，stop_info是返回错误信息
                early_stop_flag, stop_info = early_stop(trajectory, max_steps, early_stop_thresholds)

                # 判断是早停还是执行动作，如果早停就返回停止动作，如果可以就输入任务，历史，返回一个需要执行的动作
                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        # 组装prompt，转化成输入格式，得到输出，解析答案
                        action = agent.next_action(args=args, trajectory=trajectory, intent=intent, meta_data=meta_data)
                    except ValueError as e:
                        action = create_stop_action(f"ERROR: {str(e)}")
                trajectory.append(action)

                # 把字典形式的动作转化为字符串
                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None)
                render_helper.render(action, state_info, meta_data, args.render_screenshot)
                meta_data["action_history"].append(action_str)
                # 如果动作是停止，那么就直接停止
                if action["action_type"] == ActionTypes.STOP:
                    break

                # 输入动作，环境返回观测，返回观测，是否终止(始终是否)和所有信息
                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            # 一轮完成以后进行评测
            evaluator = evaluator_router(config_file)
            score = evaluator(
                trajectory=trajectory,
                config_file=config_file,
                page=env.page,
                client=env.get_page_client(env.page),
            )

            scores.append(score)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.save_trace_enabled:
                env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")

        except openai.error.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    env.close()
    logger.info(f"Average score: {sum(scores) / len(scores)}")


def early_stop(
        trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to early stop"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    # 从第一个开始取，每隔一个取出，直到最后一个，然后把后三个取出来
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
                [
                    action["action_type"] == ActionTypes.NONE
                    for action in last_k_actions
                ]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all(
                    [
                        is_equivalent(action, last_action)
                        for action in last_k_actions
                    ]
            ):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if (
                sum([is_equivalent(action, last_action) for action in action_seq])
                >= k
        ):
            return True, f"Same typing action for {k} times"

    return False, ""


def init_config(config_file):
    # 用HTML的格式保存轨迹，方便可视化
    render_helper = RenderHelper(config_file, args.result_dir, args.action_set_tag)

    # get intent
    with open(config_file) as f:
        # _c加载的是config_taskid.json，就是任务的各种信息
        _c = json.load(f)
        intent = _c["intent"]
        task_id = _c["task_id"]
        # automatically login
        if _c["storage_state"]:
            cookie_file_name = os.path.basename(_c["storage_state"])  # 加载任务信息里面，原始认证数据的存储位置
            comb = get_site_comb_from_filepath(cookie_file_name)  # 获取该网站的名字
            if not os.path.exists(".auth/tmp"):
                os.makedirs(".auth/tmp")
            temp_dir = tempfile.mkdtemp(dir=".auth/tmp")  # 建立临时存储认证，然后执行认证得到新的认证，但是value和expires会发生变化，不清楚原因
            # subprocess to renew the cookie
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
            # update the config file
            config_file = f"{temp_dir}/{os.path.basename(config_file)}"  # 更新storage_state后，把原来的config_taskid.json又存储了一遍
            with open(config_file, "w") as f:
                json.dump(_c, f)

    logger.info(f"[Config file]: {config_file}")
    logger.info(f"[Intent]: {intent}")
    return intent, task_id, config_file, render_helper


if __name__ == "__main__":
    args = config()
    prepare(args)
    test_file_list = get_test_file_list()
    agent = construct_agent(args)  # 主要进行agent的各种初始化
    test(args, agent, test_file_list)
