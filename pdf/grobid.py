from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
loader = PyPDFDirectoryLoader("/Users/elahmedi/Downloads/grobid/")
docs = loader.load()
index = VectorstoreIndexCreator().from_loaders([loader])
result2 = index.query("What are the names of the first authors tested neural networks ?")
print(result2)
#from langchain.document_loaders import WebBaseLoader
#from langchain.indexes import VectorstoreIndexCreator
#loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#index = VectorstoreIndexCreator().from_loaders([loader])

#result1 = index.query("What is Task Decomposition?")
#print(result1)
