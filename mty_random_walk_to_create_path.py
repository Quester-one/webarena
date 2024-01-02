def generate_top_k_task(website, top_k):
    return []


def beam_search_top_k_path(task_list, beam_search_len, beam_search_width):
    return []


def find_similar_page(path_list, top_k):
    return []


def random_select_span(page_list, top_k):
    return []


def regenerate_question(spans_list):
    return []


def regenerate_path(question_list, spans_list):
    return []


if __name__ == "__main__":
    import os
    import google.generativeai as genai
    from config_private import http_proxy, https_proxy, GEMINI_API_KEY

    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    genai.configure(api_key=GEMINI_API_KEY)

    website = "shopping_admin"
    agent_generated_top_k_task = 20
    beam_search_len = 10
    beam_search_width = 5
    similar_page_per_path_top_k = 3
    span_per_page = 1
    generated_task_list = generate_top_k_task(website=website, top_k=agent_generated_top_k_task)
    beam_search_path_list = beam_search_top_k_path(task_list=generated_task_list, beam_search_len=beam_search_len,
                                                   beam_search_width=beam_search_width)
    page_list = find_similar_page(path_list=beam_search_path_list, top_k=similar_page_per_path_top_k)
    spans_list = random_select_span(page_list=page_list, top_k=span_per_page)
    question_list = regenerate_question(spans_list=spans_list)
    path_list = regenerate_path(question_list=question_list, spans_list=spans_list)
