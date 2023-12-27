# from openai import OpenAI
# import os
# from dotenv import load_dotenv
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = "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"
# OpenAI.api_key = os.getenv("OPENAI_API_KEY")
# client = OpenAI()

# completion = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# print(completion.choices[0].message)

from langchain.llms import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")
chat=ChatOpenAI()
messages = [
    SystemMessage(
        content="You are a helpful assistant that translates English to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    ),
]
print(chat(messages))
# print(ChatOpenAI())
