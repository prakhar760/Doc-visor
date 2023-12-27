#read_docs_from_uri (pending)
#read_docs_from_path $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#init_milvs $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#init_trade_llm
#query




from flask import Flask, request, jsonify
#imports for read_docs_from_directory util function
import traceback

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
from langchain.vectorstores.milvus import Milvus
from milvus import default_server, debug_server

import openai
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

#-------------------------------------------------------------

import os
from dotenv import load_dotenv
load_dotenv()

load_dotenv()
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

#global variables
qa=None

#To intialize llm indexing and start milvus server
@app.route('/trade_llm/init_trade_llm', methods=['POST'])
def initialize_trade_llm():
    try:
        data = request.json
        path = data['path']
        splits=read_docs_from_directory(path)
        llm=ChatOpenAI(temperature=0)
        embeddings=OpenAIEmbeddings()
        vectordb=None
        try:
            vectordb = Milvus.from_documents(
            documents=splits,
            embedding=embeddings,
            # persist_directory=persist_directory,
            connection_args={"host": "127.0.0.1", "port":default_server.listen_port}
        )
        except Exception as e:
            print(f"Error occured while connecting to Milvus: {e}")
            print(f"Stack trace:\n{traceback_str}")
            return jsonify({'error': str(e)}), 500

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        retriever=vectordb.as_retriever()
        global qa
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory
        )
        
        return jsonify({'message': 'Trade LLM Initialization Successful'})

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error occurred: {e}")
        print(f"Stack trace:\n{traceback_str}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/trade_llm/init_milvus', methods=['GET'])
def initialize_milvus():
    try:
        debug_server.start()
        
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
        if qa==None:
            raise ValueError("Trade LLM Initialization failed \n Please Initilize trade llm first \n Hint: /trade_llm/init_trade_llm")
        result = qa({"question": query})
        print(result['answer'])
        # "Compare and contrast the airports in Seattle, Houston, and Toronto. "
        # print(str(response_chatgpt))
        
        return jsonify({'message': 'Success','queryresut':str(result['answer'])})

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

    documents=os.listdir(path)

    docs = []

    for doc_name in documents:
        doc=PyPDFLoader(f"{path}\\{doc_name}")
        docs.extend(doc.load())
        with open(f"documents\\{doc_name}.txt",'w',encoding="utf-8") as f:
            for page in doc.load():
                f.write(page.page_content)
    # Create an instance of TextSplitter
    text_splitter = RecursiveCharacterTextSplitter(seperators=["\n\n", "\n","(?<=\. )"," ", ""],chunk_size=1500, chunk_overlap=150)
    # Call the split_documents method on the instance
    docs = text_splitter.split_documents(docs)
    return docs

def read_docs_from_web(path):
    #ready to query over all of these documents.
    pass


if __name__ == '__main__':
    app.run(port=5000, debug=True)
