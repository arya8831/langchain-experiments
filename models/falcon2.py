from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter

'''from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma'''
from langchain.chains import RetrievalQA
# from langchain.schema import retriever

from langchain.vectorstores import Chroma
import textwrap
import os
from dotenv import load_dotenv, find_dotenv
from langchain import HuggingFaceHub
from langchain.document_loaders import DirectoryLoader
from langchain import PromptTemplate, LLMChain

'''from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader'''

# --------------------------------------------------------------
# Load the HuggingFaceHub API token from the .env file
# --------------------------------------------------------------

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.environ["HUGGINGFACEHUB_API_TOKEN"]

'''txt_loader = DirectoryLoader('D:/users/desktop/clone2', glob="**/*.txt")'''

pdf_loader = DirectoryLoader('C:/Users/chatt/PycharmProjects/langchain-experiments', glob="**/*.pdf")
# readme_loader = DirectoryLoader('/content/Documents/', glob="**/*.md")
txt_loader = DirectoryLoader('C:/Users/chatt/PycharmProjects/langchain-experiments', glob="**/*.txt")

# take all the loader
loaders = [txt_loader]

# lets create document
documents = []
for loader in loaders:
    documents.extend(loader.load())

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=40)  # chunk overlap seems to work better
documents = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
vectorStore = Chroma.from_documents(documents, embeddings)
# vectorStore = FAISS.from_documents(documents, embeddings)

# --------------------------------------------------------------
# Load the LLM model from the HuggingFaceHub
# --------------------------------------------------------------

repo_id = "tiiuae/falcon-7b-instruct"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options
falcon_llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 500}
)

# --------------------------------------------------------------
# Create a PromptTemplate and LLMChain
# --------------------------------------------------------------
'''template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)'''


def askanything():
    template = """
  You are an intelligent chatbot. Help the following question with brilliant answers.
  Question: {question}
  Answer:"""
    prompt = PromptTemplate(template=template, input_variables=["question"])

    llm_chain = LLMChain(prompt=prompt, llm=falcon_llm)

    question = input("Question: ")
    response = llm_chain.run(question)

    wrapped_text = textwrap.fill(
        response, width=100, break_long_words=False, replace_whitespace=False)
    print("Answer: " + wrapped_text)


def chatwithtext():
    chain = RetrievalQA.from_chain_type(llm=falcon_llm, chain_type="stuff", retriever=vectorStore.as_retriever())

    query = input("Question: ")
    response = chain.run(query)
    wrapped_text = textwrap.fill(
        response, width=100, break_long_words=False, replace_whitespace=False)
    print("Answer: " + wrapped_text)


# --------------------------------------------------------------
# Run the LLMChain
# --------------------------------------------------------------

'''question = "how to upload a video to youtube ?"
response = llm_chain.run(question)
wrapped_text = textwrap.fill(
    response, width=500, break_long_words=False, replace_whitespace=False
)
print(wrapped_text)'''

while True:
    a = input("1. Ask any question.\n2. Ask questions about Mines act.\n3. Exit\n")
    if int(a) == 1:
        askanything()
    elif int(a) == 2:
        chatwithtext()
    elif int(a) == 3:
        print("Bye.")
        break
    else:
        print("Enter valid option.")
