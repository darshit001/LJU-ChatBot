# import os
# import streamlit as st
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import Cohere 
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate

# from dotenv import load_dotenv
# load_dotenv()
# cohere_api_key = os.environ.get("COHERE_API_KEY") 

# st.title("LJ University Chat-Bot") 

# urls = ["https://ljku.edu.in/","https://ljku.edu.in/contactus","https://ljku.edu.in/placement-team"
#         ]
# loader = UnstructuredURLLoader(urls=urls)
# data = loader.load()  

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
# docs = text_splitter.split_documents(data)
# all_splits = docs
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
# vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)


# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})


# llm = Cohere(cohere_api_key=cohere_api_key, model="command-xlarge", temperature=0.1)
# query = st.chat_input("Ask me anything: ") 

# prompt = query

# system_prompt = (
#     "You are an AI assistant designed to answer questions about LJ University "
#     "based on the provided information. Use the retrieved context to generate accurate, "
#     "concise, and helpful responses. If you don't find sufficient information in the context, "
#     "clearly state that you are unable to answer the question. Do not provide fabricated or irrelevant information. "
#     "Keep your responses under three sentences."
#     "\n\n"
#     "Context:\n{context}\n\n"
#     "Question:"
# )


# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )


# if query:
#     question_answer_chain = create_stuff_documents_chain(llm, prompt)
#     rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#     response = rag_chain.invoke({"input": query})
#     print(response["answer"])

#     st.write(response["answer"])



import os
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Cohere 
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")

# Streamlit App Title
st.title("LJ University Chat-Bot")

# Define URLs to scrape
urls = [
    "https://ljku.edu.in/",
    "https://ljku.edu.in/contactus",
    "https://ljku.edu.in/placement-team",
    "https://ljku.edu.in/institutes"
    "https://ljku.edu.in/convocation"
]
loader = UnstructuredURLLoader(urls=urls)
data = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)
all_splits = docs

# Initialize embeddings and vectorstore
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Initialize Cohere LLM
llm = Cohere(cohere_api_key=cohere_api_key, model="command-xlarge", temperature=0.1)

# System Prompt
system_prompt = (
    "You are an AI assistant designed to answer questions about LJ University "
    "based on the provided information. Use the retrieved context to generate accurate, "
    "concise, and helpful responses. If you don't find sufficient information in the context, "
    "clearly state that you are unable to answer the question. Do not provide fabricated or irrelevant information. "
    "Keep your responses under three sentences."
    "\n\n"
    "Context:\n{context}\n\n"
    "Question:"
)

# Create a chat-like prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Initialize session state for storing conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
query = st.chat_input("Ask me anything about LJ University:")

# Process user query
if query:
    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Get response
    response = rag_chain.invoke({"input": query})["answer"]

    # Store the question and answer in session state
    st.session_state.chat_history.append({"question": query, "answer": response})

# Display chat history
if st.session_state.chat_history:
    for i, chat in enumerate(st.session_state.chat_history):
        st.write(f"**You:** {chat['question']}")
        st.write(f"**Bot:** {chat['answer']}")
