if __name__ == "__main__":
    import os
    import google.generativeai as genai
    from config_private import http_proxy, https_proxy, GEMINI_API_KEY
    from PIL import Image
    import matplotlib.pyplot as plt

    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    genai.configure(api_key=GEMINI_API_KEY)

    model_v = genai.GenerativeModel('gemini-pro-vision')
    model_t = genai.GenerativeModel('gemini-pro')
    image = Image.open('image_data/website_example.png')
    # plt.figure(figsize=(13, 13))
    # plt.imshow(image)
    # plt.axis('off')
    # plt.show()

    caption_prompt = "Please generate a detailed description for this image"
    caption_message = [image, caption_prompt]
    caption_result = model_v.generate_content(caption_message).text

    ocr_prompt = "Please use a table to output the text in the picture"
    ocr_message = [image, ocr_prompt]
    ocr_result = model_v.generate_content(ocr_message).text

    qa_pair_per_page_num = 5
    answer_prompt = "Please choose {} words with the clearest meaning, just output this word,and separated by ***".format(
        qa_pair_per_page_num)
    answer_messages = [{'role': 'user', 'parts': [answer_prompt + ocr_result]}]
    answer_response = model_t.generate_content(answer_messages).text
    answer_list =  list(filter(None, answer_response.split('*')))

    qa_dict={}
    for answer in answer_list:
        question_prompt = "Suppose you are the owner of a store. Please generate a natural language question to use {} as answer based on the picture.".format(
            answer)
        question_message = [image, question_prompt]
        question_result = model_v.generate_content(question_message).text
        qa_dict[question_result]=answer

    print("end")
