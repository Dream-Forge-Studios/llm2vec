# -*- coding: utf-8 -*-

"""
Install the Google AI Python SDK

$ pip install google-generativeai

See the getting started guide for more information:
https://ai.google.dev/gemini-api/docs/get-started/python
"""

import os

import google.generativeai as genai
import json

# genai.configure(api_key=os.environ["GEMINI_API_KEY"])


# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config_plain = {
  "temperature": 0.2,
  "top_p": 0.8,
  "top_k": 20,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
generation_config_json = {
  "temperature": 0.2,
  "top_p": 0.8,
  "top_k": 20,
  "max_output_tokens": 8192,
  "response_mime_type": "text/json",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model_plain = genai.GenerativeModel(
  model_name="gemini-1.5-pro-latest",
  safety_settings=safety_settings,
  generation_config=generation_config_plain,
  system_instruction="정확한 답변을 제공합니다.",
)

model_json = genai.GenerativeModel(
  model_name="gemini-1.5-pro-latest",
  safety_settings=safety_settings,
  generation_config=generation_config_json,
  system_instruction="정확한 답변을 제공합니다.",
)

contextPrompt = """
판결요지를 조건에 맞게 설명해줘

조건

1. 법률이 명시되어 있으면 반드시 포함할 것
2. 일반인이 보아도 쉽게 이해될 수 있도록 설명할 것
3. 제목, 상황, 판결 요지, 쉽게 이해하기, 결론, 핵심 법률 순으로 작성할 것
"""
qaPrompt = """
위의 내용을 기반으로 답할 수 있는 일상 생활에서 일어날 수 있는 질문(q)과 답변(a)을 10가지만 만들어줘
"""
with open('D:\llm2vec\lawVec_trainData.json', 'r', encoding='utf-8') as f:
  results = json.load(f)

n = len(results)
print(n)
for result in results[:10]:
    if len(result['summary'].strip()) >= 200:
      concatContent = '판결요지: ' + result['summary'].strip() + '\n' + contextPrompt
      chat_session = model_plain.start_chat(
        history=[]
      )
      response = chat_session.send_message(concatContent)
      context = response.text
      print(context)
      # chat_session = model.start_chat(
      #   history=[]
      # )
      #
      # print(response.text)
    else:
      n -= 1
print(n)




chat_session = model.start_chat(
  history=chat_session.history
)

print(response.text)
# print(chat_session.history)

response = chat_session.send_message("""
작성해준 설명을 기반으로 답할 수 있는 일상 생활에서 일어날 수 있는 질문과 답변을 10가지만 만들어줘
""")

print(response.text)