# Step 1 Load
import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

loader = PyPDFDirectoryLoader("/Users/elahmedi/Downloads/grobid/ped-surg-ai/pdf")
docs = loader.load()

# Step 2 Split

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 200)
all_splits = text_splitter.split_documents(docs)

# Step 3 Store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Step 4 Retrieve
question = input("enter your question please: ")
docs = vectorstore.similarity_search(question)
#len(docs)

# Step 5 Generate
# from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chat_models import ChatOpenAI
#
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=vectorstore.as_retriever())
#result = qa_chain({"query": question})
#print(result)


llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
print(qa_chain({"query": question+" at the end of the answer, cite the referenced paper in Vancouver style. If unsure, say so."},return_only_outputs=True))

# Step 6 Memory
# Will be added in the future