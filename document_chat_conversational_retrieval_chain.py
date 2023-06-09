import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import GPT4All, HuggingFaceHub
from streamlit_chat import message
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_loaders import PyPDFLoader
import os

def extract_text_chunks(doc):
    text = ""
    pdf_reader = PdfReader(doc)        
    for page in pdf_reader.pages:
        # read and concatenate the text from
        # each page into the raw text string
        text += page.extract_text()
        
    return get_text_chunks(text)

    

def get_text_chunks(raw_text):
    # return text chunks from the raw text extracted
    # from the user PDFs
    # st.info("Creating text chunks from raw text")
    # initializing text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len
    )
    # creating text chunks
    chunks = text_splitter.split_text(raw_text)
    # st.success("Text chunks created")
    return chunks


def handle_user_input(user_question):
    if user_question is None:
        return
    # Handle user Queries
    print(f"Generating response to user query: {user_question}")
    response = st.session_state.conversation({'question': user_question})
    
    print("Response generated. Updating session state chat_history")
    st.session_state.chat_history = response['chat_history']
    
    for i, msg in enumerate(st.session_state.chat_history):
        is_user_message = (i%2==0)
        message(msg.content, is_user=is_user_message)

def process_new_uploads(pdf_docs):
    vectordb = st.session_state.vectorstore
    # vectordb = Chroma(persist_directory="./documents_cache/conversation_retrieval", embedding_function=embeddings)
    for doc in pdf_docs:
        print(f"Processing new file: {doc.name}")
        
        print(f"Saving original file copy: {doc.name}")
        with open(os.path.join("original_documents",doc.name),"wb") as f:
            f.write(doc.getbuffer())
        
        loader = PyPDFLoader(file_path=f"./original_documents/{doc.name}")        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)        
        text_chunks = text_splitter.split_documents(loader.load())
        
        vectordb.add_documents(documents=text_chunks)
        vectordb.persist()
    
        
        
def load_vector_database():
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", cache_folder="./model_cache")
    st.session_state.vectorstore = Chroma(persist_directory="./documents_cache/conversation_retrieval", embedding_function=embeddings)

def initialize_conversation_chain():
    callbacks = [StreamingStdOutCallbackHandler()]
    # Snoozy Model failed to start on my machine # local_model_snoozy = "./model_cache/ggml-gpt4all-l13b-snoozy.bin"
    local_model_groovy = "./model_cache/ggml-gpt4all-j-v1.3-groovy.bin"
    # HuggingFaceHub(verbose=True,cache=True,task="text-generation",repo_id="tiiuae/falcon-40b-instruct")
    llm_instance = GPT4All(model=local_model_groovy, callbacks=callbacks, verbose=True)
    vectordb = st.session_state.vectorstore
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_instance,
        retriever=vectordb.as_retriever(),
        memory=memory
    )    
    
    st.session_state.conversation = conversation_chain
    
            

def main():
    load_dotenv()
    st.set_page_config(page_title="Converse with your Documents", page_icon=":books:")
    
    st.header("Converse with your Documentation..!! :books:")
    
    if "vectorstore" not in st.session_state:
        with st.spinner("Loading Vector Database..."):
            load_vector_database()
    
    if "conversation" not in st.session_state:
        with st.spinner("Initializing AI. Please wait..."):
            initialize_conversation_chain()
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    user_question = st.text_input("Ask a question to your documents here:")
    
    if user_question:
        handle_user_input(user_question)
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
