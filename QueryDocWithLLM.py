#Setting Up Your Jupyter Notebook for Multiple Document Querying

#!pip install llama-index nltk milvus pymilvus langchain python-dotenv openai
#pip install llama-index==0.6.2
#pip install langchain==0.0.154

import nltk
import ssl

import collections.abc

# add attributes to `collections` module
# before you import the package that causes the issue
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

try:

   _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
   pass
else:

   ssl._create_default_https_context = _create_unverified_https_context

nltk.download("stopwords")

print("prakhar1")

#Building Your Document Query Engine with LlamaIndex

from llama_index import (
   GPTVectorStoreIndex,
   GPTSimpleKeywordTableIndex,
   SimpleDirectoryReader,
   LLMPredictor,
   ServiceContext,
   StorageContext
)

print("prakhar2")

# from langchain.llms import OpenAIChat
from langchain.chat_models import ChatOpenAI

from langchain.chains import LLMChain

from langchain.llms import openai
# from langchain.memory import VectorStoreRetrieverMemory
# from langchain.chains import ConversationChain
# from langchain.prompts import PromptTemplate
# from langchain.vectorstores import Milvus

print("prakhar3")

import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"

openai.api_key = os.getenv("OPENAI_API_KEY")

#Starting the Vector Database
from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server,debug_server
print("prakhar4")

debug_server.start()
vector_store = MilvusVectorStore(
   dim=1536,
  host = "127.0.0.1",
  port = default_server.listen_port
)

print("prakhar5")

#Gathering Documents
import sys

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8') 

wiki_titles = ["Toronto", "Seattle", "San Francisco", "Chicago", "Boston", 
"Washington, D.C.", "Cambridge, Massachusetts", "Houston"]

print("prakhar6")

from pathlib import Path

import requests

for title in wiki_titles:
   response = requests.get(
       'https://en.wikipedia.org/w/api.php',
       params={
           'action': 'query',
           'format': 'json',
           'titles': title,
           'prop': 'extracts',
           'explaintext': True,
       }
).json()
   page = next(iter(response['query']['pages'].values()))
   wiki_text = page['extract']

   data_path = Path('data')
   if not data_path.exists():
       Path.mkdir(data_path)

   with open(data_path / f"{title}.txt", 'w', encoding="utf-8") as fp:
       fp.write(wiki_text)

print("prakhar6")

#Creating Your Document Indices in LlamaIndex
# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
   city_docs[wiki_title] = SimpleDirectoryReader(input_files=[f"data/{wiki_title}.txt"]).load_data()

print("prakhar7")

#create two empty dictionaries for the city indices and the summaries
llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
print("prakhar_iterator")
# Build city document index
city_indices = {}
index_summaries = {}
for wiki_title in wiki_titles:
   print("prakhar_iterator1")
   city_indices[wiki_title] = GPTVectorStoreIndex.from_documents(city_docs[wiki_title], 
service_context=service_context, storage_context=storage_context,show_progress=True)
   print("prakhar_iterator2")
   # set summary text for city
   index_summaries[wiki_title] = f"Wikipedia articles about {wiki_title}"
   print("prakhar_iterator3")


print("prakhar8")

#Decomposable Querying Over Your Documents
from llama_index.indices.composability import ComposableGraph
graph = ComposableGraph.from_indices(
   GPTSimpleKeywordTableIndex,
   [index for _, index in city_indices.items()],
   [summary for _, summary in index_summaries.items()],
   max_keywords_per_chunk=50
)
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform
decompose_transform = DecomposeQueryTransform(
   llm_predictor_chatgpt, verbose=True
)

print("prakhar9")

#create the engines to do the query transformation.
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
custom_query_engines = {}
for index in city_indices.values():
   query_engine = index.as_query_engine(service_context=service_context)
   transform_extra_info = {'index_summary': index.index_struct.summary}
# //   tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, 
# //                                 transform_extra_info=transform_extra_info)  
   tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, 
                                 transform_metadata=transform_extra_info)  
   # tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform)
   custom_query_engines[index.index_id] = tranformed_query_engine
print("prakhar10")
#query engine from the graph that uses the dictionary of custom query engines we created above.
custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
   retriever_mode='simple',
   response_mode='tree_summarize',
   service_context=service_context
)
print("prakhar11")
query_engine_decompose = graph.as_query_engine(
   custom_query_engines=custom_query_engines,)

#ready to query over all of these documents.
print("prakhar12")

response_chatgpt = query_engine_decompose.query(
   "Compare and contrast the airports in Seattle, Houston, and Toronto. "
)
print(str(response_chatgpt))
