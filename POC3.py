import os
import openai
from dotenv import load_dotenv
from pymilvus import connections, FieldSchema, CollectionSchema, DataType
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from llama_index.vector_stores import MilvusVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# # ----mapping-import-error-----------------------------------
# import collections.abc

# # add attributes to `collections` module
# # before you import the package that causes the issue
# collections.Mapping = collections.abc.Mapping
# collections.MutableMapping = collections.abc.MutableMapping
# collections.Iterable = collections.abc.Iterable
# collections.MutableSet = collections.abc.MutableSet
# collections.Callable = collections.abc.Callable

# # -----------------------------------------------------------

load_dotenv()
os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load Milvus connection
connections.connect()

# Set Milvus parameters
milvus_params = {
    "host": "127.0.0.1",
    "port": 19530,
}

# Define Milvus collection
milvus_collection_name = "tradvisor"
field_name = "embedding"
dimension = 768  # Adjust the dimension based on your embedding size
collection_schema = CollectionSchema(fields=[
    FieldSchema(name=field_name, data_type=DataType.FLOAT_VECTOR, dim=dimension)
])

# Create Milvus collection if not exists
if not connections.get_connection().has_collection(milvus_collection_name):
    connections.get_connection().create_collection(collection_schema, milvus_collection_name)

# Set Milvus vector store
vectordb = MilvusVectorStore(
    collection_name=milvus_collection_name,
    field_name=field_name,
    dimension=dimension
)

# Load OpenAI Embeddings
embedding = OpenAIEmbeddings()

# Set up ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Set up text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Set up ConversationalRetrievalChain
qa = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectordb.as_retriever(),
    memory=memory
)

# User interaction loop
question = ""
while question != "q":
    question = input("User- ")
    result = qa({"question": question})
    print(result['answer'])
