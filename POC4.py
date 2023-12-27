import os
import openai
import sys
import collections.abc
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

def setup_environment():
    sys.path.append('../..')
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
    openai.api_key = os.getenv("OPENAI_API_KEY")

def add_collections_attributes():
    # add attributes to `collections` module
    # before you import the package that causes the issue
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
    collections.Iterable = collections.abc.Iterable
    collections.MutableSet = collections.abc.MutableSet
    collections.Callable = collections.abc.Callable

def load_documents():
    documents_path = "C:\\Users\\txl_rishabh\\Documents\\documents"
    documents = os.listdir(documents_path)
    docs = []

    for doc_name in documents:
        doc = PyPDFLoader(os.path.join(documents_path, doc_name))
        docs.extend(doc.load())
        with open(f"documents\\{doc_name}.txt", 'w', encoding="utf-8") as f:
            for page in doc.load():
                f.write(page.page_content)
    return docs

def setup_openai_embedding():
    persist_directory = 'docs/chroma/'
    embedding = OpenAIEmbeddings()
    return persist_directory, embedding

def setup_chat_model():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    return splits

def setup_chroma_vector_db(splits, persist_directory, embedding):
    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectordb

def setup_conversation_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def setup_self_query_retriever(llm, vectordb, document_content_description, metadata_field_info):
    retriever = SelfQueryRetriever.from_llm(llm, vectordb, document_content_description, metadata_field_info)
    return retriever

def setup_conversational_retrieval_chain(llm, retriever, memory):
    qa = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory)
    return qa

def main():
    setup_environment()
    add_collections_attributes()

    docs = load_documents()
    llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0)


    persist_directory, embedding = setup_openai_embedding()
    vectordb = setup_chroma_vector_db(split_documents(docs), persist_directory, embedding)

    memory = setup_conversation_memory()

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The Bill can be one of the ['C:\\Users\\txl_rishabh\\Documents\\documents\\Finance_Bill.pdf'], The Research paper is ['C:\\Users\\txl_rishabh\\Documents\\documents\\I2 RESEARCH PAPER.pdf']",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The Page from the source",
            type="string"
        )
    ]

    retriever = setup_self_query_retriever(llm, vectordb, "", metadata_field_info)
    qa = setup_conversational_retrieval_chain(llm, retriever, memory)

    question = ""
    while question != "q":
        question = input("User- ")
        result = qa({"question": question})
        print("TRADVISOR- ", result['answer'], "\n")

if __name__ == "__main__":
    main()
