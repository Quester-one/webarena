if __name__ == "__main__":
    import os
    import google.generativeai as genai
    from config_private import http_proxy, https_proxy, GEMINI_API_KEY
    from PIL import Image
    import matplotlib.pyplot as plt

    os.environ["http_proxy"] = http_proxy
    os.environ["https_proxy"] = https_proxy
    genai.configure(api_key=GEMINI_API_KEY)

    model = genai.GenerativeModel('gemini-pro-vision')
    image = Image.open('image_data/website_example.png')
    plt.figure(figsize=(13, 13))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    prompt_list = [
        "Please generate a detailed description for this image",
        "Please first analyze which parts this webpage is divided into and analyze the functions of each part",
    ]

    result_dict = {}
    for prompt in prompt_list:
        message = [image, prompt]
        response = model.generate_content(message)
        result_dict[prompt] = response.text
    print("end")
