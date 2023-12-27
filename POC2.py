import os
import openai
import sys
sys.path.append('../..')

# ----mapping-import-error-----------------------------------
import collections.abc

# add attributes to `collections` module
# before you import the package that causes the issue
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

# -----------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

# from langchain.vectorstores import chroma
from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
from langchain.document_loaders import PyPDFLoader
# from langchain.vectorstores.chroma import Chroma

from langchain.vectorstores.milvus import Milvus
# from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server, debug_server

# Load PDF
# loaders = [
#     # Duplicate documents on purpose - messy data
#     PyPDFLoader("C:\\Users\\txl_rishabh\\Downloads\\Finance_Bill.pdf")
# ]
# debug_server.cleanup()
default_server.start()

documents=os.listdir("C:\\Users\\txl_rishabh\\Documents\\documents")

docs = []

for doc_name in documents:
    doc=PyPDFLoader(f"C:\\Users\\txl_rishabh\\Documents\\documents\\{doc_name}")
    docs.extend(doc.load())
    with open(f"documents\\{doc_name}.txt",'w',encoding="utf-8") as f:
        for page in doc.load():
            f.write(page.page_content)

# for loader in loaders:
#     temp=loader.load()
#     docs.extend(loader.load())
#     with open('financial.txt','w',encoding='utf-8') as f:
#         for page in temp:
#             print(page.page_content)
#             f.write(page.page_content)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)
# print(llm.predict("Hello world!"))

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)

splits = text_splitter.split_documents(docs)

#vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
vectordb = Milvus.from_documents(
    documents=splits,
    embedding=embedding,
    # persist_directory=persist_directory,
    connection_args={"host": "127.0.0.1", "port":default_server.listen_port}
)

# Build prompt
# from langchain.prompts import PromptTemplate
# template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
# {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

# Run chain
# from langchain.chains import RetrievalQA
# question = "Is probability a class topic?"
# qa_chain = RetrievalQA.from_chain_type(llm,
#                                        retriever=vectordb.as_retriever(),
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


# result = qa_chain({"query": question})
# print(result["result"])

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory
)

question = ""
while (question!="q"):
    question = input("User- ")
    result = qa({"question": question})
    print(result['answer'])

# result['answer']

# question = "why are those prerequesites needed?"
# result = qa({"question": question})
