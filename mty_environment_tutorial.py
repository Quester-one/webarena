def check_configuration():
    assert os.path.exists("config_files/0.json")
    with open("config_files/0.json", "r") as f:
        config = json.load(f)
        assert os.environ["SHOPPING_ADMIN"] in config["start_url"], (os.environ["SHOPPING_ADMIN"], config["start_url"])


def display_image(img_obs):
    plt.figure(figsize=(13, 13))
    plt.imshow(img_obs)
    plt.axis('off')
    plt.show()


def accessibility_tree_interaction():
    env = ScriptBrowserEnv(
        headless=True,
        slow_mo=100,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 1280},
    )
    config_file = "config_files/156.json"
    trajectory = []

    obs, info = env.reset(options={"config_file": config_file})
    actree_obs = obs["text"]
    img_obs = obs["image"]
    display_image(img_obs)
    state_info = {"observation": obs, "info": info}
    trajectory.append(state_info)
    match = re.search(r"\[(\d+)\] link 'Merge requests'", actree_obs).group(1)
    click_action = create_id_based_action(f"click [{match}]")
    trajectory.append(click_action)

    obs, _, terminated, _, info = env.step(click_action)
    actree_obs = obs["text"]
    img_obs = obs["image"]
    display_image(img_obs)
    state_info = {"observation": obs, "info": info}
    trajectory.append(state_info)
    match = re.search(r"\[(\d+)\] link 'Assigned to you", actree_obs).group(1)
    click_action = create_id_based_action(f"click [{match}]")
    trajectory.append(click_action)

    obs, _, terminated, _, info = env.step(click_action)
    actree_obs = obs["text"]
    img_obs = obs["image"]
    display_image(img_obs)
    state_info = {"observation": obs, "info": info}
    trajectory.append(state_info)
    trajectory.append(create_stop_action(""))

    evaluator = evaluator_router(config_file)
    score = evaluator(
        trajectory=trajectory,
        config_file=config_file,
        page=env.page,
        client=env.get_page_client(env.page),
    )

    if score == 1.0:
        print("Succeed!")


if __name__ == "__main__":
    import json
    import os
    import re
    import subprocess
    import matplotlib.pyplot as plt
    import time
    from config_private import SHOPPING, SHOPPING_ADMIN, REDDIT, GITLAB, MAP, WIKIPEDIA, HOMEPAGE, http_proxy, \
        https_proxy, OPENAI_API_KEY, proxy_server, proxy_username, proxy_password

    SLEEP = 1.5
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

    from browser_env import (
        Action,
        ActionTypes,
        ObservationMetadata,
        ScriptBrowserEnv,
        StateInfo,
        Trajectory,
        action2str,
        create_id_based_action,
        create_stop_action,
    )
    from evaluation_harness.evaluators import evaluator_router

    check_configuration()
    accessibility_tree_interaction()
