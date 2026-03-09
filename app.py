import os
import pickle
import time
import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_classic.chains import RetrievalQA
from langchain_classic.chains import RetrievalQAWithSourcesChain

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
st.title("News Research Tool")
st.sidebar.title("News Article URLS")

urls=[]

file_path = "vector_index.pkl"

for i in range(3):
    url=st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URL's")


main_placeholder=st.empty()

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.9,
    max_tokens=500
)

if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading Started...")
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n'],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter Started...")
    docs=text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorindex = FAISS.from_documents(docs, embeddings)
    
    main_placeholder.text("Embedding Vector Started Building...")
    
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex, f)
        
query=main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb")as f:
            vectorstore=pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff"
            )
            result=chain({"question":query},return_only_outputs=True)
            st.header("Answer")
            st.subheader(result["answer"])
            
            
            sources=result.get("sources","")
            if sources:
                st.subheader("Sources:")
                sources_list=sources.split("\n")
                for source in sources_list:
                    st.write(source)
        