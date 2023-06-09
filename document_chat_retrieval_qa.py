import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub, GPT4All
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader

import os
import logging as log

log.basicConfig(filename="logs/app.log", level=log.DEBUG)

def load_vector_database():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        cache_folder="./model_cache",
        model_kwargs={'device':'cuda'},
        encode_kwargs={'normalize_embeddings':False}
    )
    log.info("Initializing Vector DB")
    st.session_state.vectordb = Chroma(persist_directory="./documents_cache/qa_retrieval", embedding_function=embeddings)

def get_local_gpt4all_models():
    local_models = {}
    local_models["ggml-gpt4all-j-v1.3-groovy"] = "./model_cache/ggml-gpt4all-j-v1.3-groovy.bin"
    local_models["ggml-mpt-7b-instruct"] = "./model_cache/ggml-mpt-7b-instruct.bin"
    local_models["ggml-gpt4all-l13b-snoozy"] = "./model_cache/ggml-gpt4all-l13b-snoozy.bin"
    local_models["ggml-v3-13b-hermes-q5_1"] = "./model_cache/ggml-v3-13b-hermes-q5_1.bin"
    local_models["ggml-vicuna-13b-1.1-q4_2"] = "./model_cache/ggml-vicuna-13b-1.1-q4_2.bin"
    
    return local_models


def initialize_conversation_chain():
    vectordb = st.session_state.vectordb
    callbacks = [StreamingStdOutCallbackHandler()]
    local_models = get_local_gpt4all_models()
    
    llm_instance = GPT4All(
        model=local_models["ggml-gpt4all-j-v1.3-groovy"], 
        callbacks=callbacks, 
        verbose=True,
        # n_ctx=2048,
        # temp=0.1
    )

    # llm_instance = ChatOpenAI(temperature=0.1)
    
    # llm_instance = HuggingFaceHub(
    #     verbose=True,
    #     task="text-generation",
    #     repo_id="tiiuae/falcon-40b-instruct"
    # )
    
    retriever_instance = vectordb.as_retriever(search_kwargs={'k':2})
    
    log.info("Initializing QA Chain")
    st.session_state.qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=retriever_instance
    )
    

def handle_user_input(user_question, response_container):
    if user_question is None:
        return
    
    qa_chain:RetrievalQAWithSourcesChain = st.session_state.qa_chain
    
    response_container.empty()
    # Handle user Queries
    with response_container.container():
        with st.spinner("Generating Response..."):
            log.info(f"Generating response to user query: {user_question}")
            response = qa_chain({"question":user_question}, return_only_outputs=True)
        
            # st.write(response)                
            st.write(response["answer"])
            
            with st.expander(label="Sources", expanded=False):
                for source in response["sources"]:
                    st.write(source)
    


def process_new_uploads(pdf_docs):
    vectordb:Chroma = st.session_state.vectordb
    for doc in pdf_docs:
        log.info(f"Processing new file: {doc.name}")
        
        with open(os.path.join("tmp_documents",doc.name),"wb") as f:
            f.write(doc.getbuffer())
        
        loader = PyPDFLoader(file_path=f"./tmp_documents/{doc.name}")        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=640, chunk_overlap=128)        
        
        log.info("Creating text chunks")
        text_chunks = text_splitter.split_documents(loader.load())
        # log.info("Chunks: %s", text_chunks)
        
        log.info("Run embeddings and add document to Vector DB")
        vectordb.add_documents(documents=text_chunks)
        vectordb.persist()        
        os.remove(f"./tmp_documents/{doc.name}")
        log.info(f"File processed successfully: {doc.name}")
    

def main():
    load_dotenv()
    st.set_page_config(page_title="Converse with your Documents", page_icon=":books:")
    
    st.header("Converse with your Documentation..!! :books:")
    
    if "vectordb" not in st.session_state:
        with st.spinner("Initializing Vector DB..."):
            load_vector_database()
    
    if "qa_chain" not in st.session_state:
        with st.spinner("Initializing AI Model..."):
            initialize_conversation_chain()
    
    user_question = st.text_input("Ask a question to your documents here:")
    
    response_container = st.empty()
    
    if user_question:
        handle_user_input(user_question, response_container)
        user_question = None

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing..."):                
                process_new_uploads(pdf_docs)


if __name__ == "__main__":
    main()
