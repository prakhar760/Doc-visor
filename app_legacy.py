from flask import Flask, request, jsonify
#imports for read_docs_from_directory util function

# ----mapping-import-error-----------------------------------
import collections.abc

# add attributes to `collections` module
# before you import the package that causes the issue
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

# ------------------------------------------------------------

import traceback
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter

# ------------------------------------------------------------

#Starting the Vector Database
import os
from dotenv import load_dotenv

from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server,debug_server

#imports for document query engine
from llama_index import (
   GPTVectorStoreIndex,
   GPTSimpleKeywordTableIndex,
   SimpleDirectoryReader,
   LLMPredictor,
   ServiceContext,
   StorageContext
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.llms import openai

from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.indices.composability import ComposableGraph
from llama_index.indices.query.query_transform.base import DecomposeQueryTransform

load_dotenv()
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

#global variables
vector_store=None #check while making queries and initializing trade llm
query_engine_decompose=None

#To intialize llm indexing and start milvus server
@app.route('/trade_llm/init_trade_llm', methods=['POST'])
def initialize_trade_llm():
    try:
        data = request.json
        path = data['path']
        docs=read_docs_from_directory(path)
        print(docs,'#\n#\n#\n#\n#\n')
        #create a dictionary with file names in it currrently leaving it blank
        wiki_titles={'Visual Speech Synthesis System Using Machine Learning'}
        #create a dictionary to pass into gptvectorstoreindex
        city_docs = {}
        for wiki_title in wiki_titles:
            city_docs[wiki_title] = SimpleDirectoryReader(input_files=[f"document.txt"]).load_data()

        #create two empty dictionaries for the city indices and the summaries
        llm_predictor_chatgpt = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor_chatgpt)
        if vector_store==None:
            raise ValueError("Trade LLM Initialization failed \n Please Initilize milvus first \n Hint: /trade_llm/init_milvus")
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # Build city document index
        city_indices = {}
        index_summaries = {}
        for wiki_title in wiki_titles:
            city_indices[wiki_title] = GPTVectorStoreIndex.from_documents(city_docs[wiki_title], 
        service_context=service_context, storage_context=storage_context)
        # set summary text for city
            index_summaries[wiki_title] = f"Machine Learning Paper about {wiki_title}"
        
        #Decomposable Querying Over Your Documents
        graph = ComposableGraph.from_indices(
        GPTSimpleKeywordTableIndex,
        [index for _, index in city_indices.items()],
        [summary for _, summary in index_summaries.items()],
        max_keywords_per_chunk=50
        )
        decompose_transform = DecomposeQueryTransform(
        llm_predictor_chatgpt, verbose=True
        )


        #create the engines to do the query transformation.
        custom_query_engines = {}
        for index in city_indices.values():
            query_engine = index.as_query_engine(service_context=service_context)
            transform_extra_info = {'index_summary': index.index_struct.summary} 
            tranformed_query_engine = TransformQueryEngine(query_engine, decompose_transform, 
                                        transform_metadata=transform_extra_info)  
            custom_query_engines[index.index_id] = tranformed_query_engine

        #query engine from the graph that uses the dictionary of custom query engines we created above.
        custom_query_engines[graph.root_index.index_id] = graph.root_index.as_query_engine(
        retriever_mode='simple',
        response_mode='tree_summarize',
        service_context=service_context
        )
        global query_engine_decompose
        query_engine_decompose = graph.as_query_engine(
        custom_query_engines=custom_query_engines,)

        
        return jsonify({'message': 'Trade LLM Initialization Successful'})

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error occurred: {e}")
        print(f"Stack trace:\n{traceback_str}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/trade_llm/init_milvus', methods=['GET'])
def initialize_milvus():
    try:
        global vector_store
        debug_server.start()
        vector_store = MilvusVectorStore(
        host = "127.0.0.1",
        port = default_server.listen_port
        )
        
        
        return jsonify({'message': 'Initialization successful'})

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error occurred: {e}")
        print(f"Stack trace:\n{traceback_str}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/trade_llm/query', methods=['POST'])
def query():
    try:
        data = request.json
        query=data['query']
        if vector_store==None or query_engine_decompose==None:
            raise ValueError("Trade LLM Initialization failed \n Please Initilize trade llm first \n Hint: /trade_llm/init_trade_llm")
        response_chatgpt = query_engine_decompose.query(
            query
        )
        # "Compare and contrast the airports in Seattle, Houston, and Toronto. "
        print(str(response_chatgpt))
        
        return jsonify({'message': 'Success','queryresut':str(response_chatgpt)})

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error occurred: {e}")
        print(f"Stack trace:\n{traceback_str}")
        return jsonify({'error': str(e)}), 500
    
#Done
def read_docs_from_directory(path: str) -> str:
    ###
    #input: expects a link
    #output: returns the content of pdf in text
    ###

    # Create an instance of PyPDFLoader
    loader = PyPDFLoader(path,extract_images=True)  
    # Load and split the document
    pages = loader.load_and_split()
    print(len(pages))
    # Create an instance of TextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # Call the split_documents method on the instance
    docs = text_splitter.split_documents(pages)

    #Legacy
    document_text=""
    for page in docs:
        delimeter=''
        # if(page.page_content[-1]!='.'):
        #     delimeter='\n'
        document_text+=delimeter+page.page_content

    # with open("document.txt", 'w', encoding="utf-8") as fp:
    #    fp.write(document_text)
    # print(document_text)
    return document_text
def read_docs_from_web(path):
    #ready to query over all of these documents.
    pass


if __name__ == '__main__':
    app.run(port=5000, debug=True)
