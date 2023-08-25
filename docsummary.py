from langchain.chains.mapreduce import MapReduceChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ReduceDocumentsChain, MapReduceDocumentsChain
from langchain.document_loaders import PyPDFDirectoryLoader #needed
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


loader = PyPDFDirectoryLoader("/Users/elahmedi/Downloads/grobid/ped-surg-ai/surgsystems")
docs = loader.load()
print("the number of loaded pages is: ",len(docs))
#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm = ChatOpenAI(temperature=0.1, model_name="gpt-4")
# Map
map_template = """The following is a set of documents
{docs}
Based on this list of documents, please identify if and how authors reduced bias and class imbalance, and diversified their training data. Use in-text citations to cite each study.
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)

# Reduce
reduce_template = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final thesis discussion that ties with the importance of upholding principles of equity, diversity, and inclusivity in AI. Add in-text citation for each included study.
Helpful Answer:"""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="doc_summaries"
)

# Combines and iteravely reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=8000,
)

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 200)
split_docs = text_splitter.split_documents(docs)

print(map_reduce_chain.run(split_docs))