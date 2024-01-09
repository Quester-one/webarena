import json


def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {file_path}")
        return None


if __name__ == "__main__":
    '''
    参数说明
    cot/direct:有无思维链
    2s/3s:few-shot数量
    None/no_na:no_na表示可以认为任务无法完成
    None/llama:llama版本的原则约束更加简短
    
    内容说明
    intro:原则+动作元素
    examples:list，每个元素是一个example。一个example里面由两部分构成，第一部分里面的元素是少数几个观测，任务描述，历史动作None，url，第二部分是思维链+结果```stop [结果]```
    template:模板，'OBSERVATION:{observation} URL: {url} OBJECTIVE: {objective} PREVIOUS ACTION: {previous_action}'
    meta_data: 一些参数，{'observation': 'accessibility_tree', 'action_type': 'id_accessibility_tree', 'keywords': ['url', 'objective', 'observation', 'previous_action'], 
                        'prompt_constructor': 'CoTPromptConstructor', 'answer_phrase': 'In summary, the next action I will perform is', 'action_splitter': '```'}
    '''
    p_cot_id_actree_2s = read_json("/data/mentianyi/code/webarena/agent/prompts/jsons/p_cot_id_actree_2s.json")
    p_cot_id_actree_2s_no_na = read_json(
        "/data/mentianyi/code/webarena/agent/prompts/jsons/p_cot_id_actree_2s_no_na.json")
    p_direct_id_actree_2s = read_json("/data/mentianyi/code/webarena/agent/prompts/jsons/p_direct_id_actree_2s.json")
    p_direct_id_actree_2s_no_na = read_json(
        "/data/mentianyi/code/webarena/agent/prompts/jsons/p_direct_id_actree_2s_no_na.json")
    p_direct_id_actree_3s_llama = read_json(
        "/data/mentianyi/code/webarena/agent/prompts/jsons/p_direct_id_actree_3s_llama.json")
    print("end")
