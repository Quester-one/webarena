import PIL.Image
import os
import google.generativeai as genai
from config_private import http_proxy, https_proxy, GEMINI_API_KEY
import numpy as np
from PIL import Image

os.environ["http_proxy"] = http_proxy
os.environ["https_proxy"] = https_proxy
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-pro-vision')
img1 = PIL.Image.open('./../image_data/dog.jpg')
img2 = PIL.Image.open('./../image_data/cat.jpg')

numpy_array = np.random.randint(0, 255, size=(100, 100, 3), dtype=np.uint8)
image = Image.fromarray(numpy_array)

message_1 = [image, img1, "lll", img2, "Give these two animals two new names and give reasons?"]
response = model.generate_content(message_1)
print(response.text)

model_text = genai.GenerativeModel('gemini-pro')
messages = [{'role': 'user', 'parts': ["Briefly explain how a computer works to a young child."]}]
response = model_text.generate_content(messages)
messages.append({'role': 'model',
                 'parts': [response.text]})
messages.append({'role': 'user',
                 'parts': ["Okay, how about a more detailed explanation to a high school student?"]})
response = model_text.generate_content(messages)
print(response.text)
