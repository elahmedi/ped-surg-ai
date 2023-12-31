# TODO
# Due to the large number of full texts,
# batch processing appears to be necessary.

# pip3 install scikit-learn

# Step 1 Load
import os
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.document_loaders import CSVLoader   #TODO: to talk to the dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
#from langchain.chains.summarize import load_summarize_chain
from langchain.retrievers import SVMRetriever

loader = PyPDFDirectoryLoader("/Users/elahmedi/Downloads/grobid/ped-surg-ai/pdf")
source_material = loader.load()
print("documents were loaded successfully")
# Step 2 Split

# text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 200)
# all_splits = text_splitter.split_documents(docs)
# print("documents were split successfully")
# Step 3 Store
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
# Step 2 Prepare retriever
svm_retriever = SVMRetriever.from_documents(source_material,OpenAIEmbeddings())
print("SVM initialized successfully")
# Step 3 Ask question
while True:
    question = input("enter your question please: ")
    creativity = input("On a scale of 0 to 1, how creative do you want the answer to be? Setting it to zero avoids hallucination. ")
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=creativity)
    llm = ChatOpenAI(model_name="gpt-4", temperature=creativity)
    docs=svm_retriever.get_relevant_documents(question)
    print("the number of relevant studies is ",len(docs))
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=svm_retriever)
    print(qa_chain({"query": question+" at the end of the answer, cite the referenced paper(s) in Vancouver style."},return_only_outputs=True))

# Step 6 Chatbot with Memory
# Will be added in the future