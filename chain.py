import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain

loader = PyPDFDirectoryLoader("/Users/elahmedi/Downloads/grobid/ped-surg-ai/pdf")
docs = loader.load()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
chain = load_summarize_chain(llm, chain_type="refine")
chain.run(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 200)
split_docs = text_splitter.split_documents(docs)

print(map_reduce_chain.run(split_docs))