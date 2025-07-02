import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from IPython.display import Markdown,display
from openai import OpenAI



load_dotenv(override=True)
api_key=os.getenv('OPENAI_API_KEY')
if not api_key:
    print('api key not found')
else:
    print('api key found')

system_prompt="""
You are a personal tutor for the user. You are helpful, friendly, and explain everything in very simple terms. 
You never assume prior knowledge unless the user explicitly says so. 
Use analogies, real-life examples, and step-by-step breakdowns.

You speak like a human mentor — calm, encouraging, and clear. 
If the user asks for a quiz, give 3-5 questions based on the current topic. 
If they get something wrong, gently correct and explain it.

Always end with: "Would you like to learn more or try a quiz?"
"""
def user_prompt_tutor(question) : 
    user_prompt=user_prompt = f"Explain the following question to me as if I'm a complete beginner: \n {question}"
    return user_prompt


openai=OpenAI()
def tutor(question):
    response=openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role':'system','content':system_prompt},
            {'role':'user','content':user_prompt_tutor(question)}
        ]
    )
    return response.choices[0].message.content

Question = input('I am your personal Tutor Please Ask me Anything :D ')

print(tutor(Question))
