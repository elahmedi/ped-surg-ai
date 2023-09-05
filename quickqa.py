from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI, PromptTemplate
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFDirectoryLoader

import glob
import os

loader = PyPDFDirectoryLoader("/Users/elahmedi/Downloads/grobid/ped-surg-ai/gs")
docs = loader.load()
print("number of pages is: ", len(docs))
index = VectorstoreIndexCreator().from_loaders([loader])
print("vector store created")
while True:
    question = input("ask: ")
    print(index.query(question, llm="gpt-4"))