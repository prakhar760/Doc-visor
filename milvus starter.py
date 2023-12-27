# ----mapping-import-error-----------------------------------
import collections.abc

# add attributes to `collections` module
# before you import the package that causes the issue
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

from langchain.vectorstores.milvus import Milvus
# from llama_index.vector_stores import MilvusVectorStore
from milvus import default_server,debug_server

from flask import Flask

app = Flask(__name__)
debug_server.start()
@app.route('/', methods=['GET'])
def initi():
    pass

if __name__ == '__main__':
    app.run(port=5000, debug=True)