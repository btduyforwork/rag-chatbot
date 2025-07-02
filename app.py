import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from InstructorEmbedding import INSTRUCTOR  # 👈 Quan trọng
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from htmlTemplate import css,bot_template,user_template


# State initialization:
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def get_pdf_text(pdf_docs: any):
    text=""
    for pdf_file in pdf_docs:
        pdf_reader=PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text 

def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vector_store(chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")  # Lưu ý model name đúng
    vector_DB = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_DB

def get_conversation_chain(vector_DB):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=ChatOllama(model="llama3"),
        retriever=vector_DB.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_question(user_question):
    response=st.session_state.conversation({"question":user_question})
    # st.write(response)

    st.session_state.chat_history=response['chat_history']

def display_chat():
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None

    st.header("Chat with multiple PDFs :books:")

    chat_container = st.container(height=400)

    user_question=st.text_input("Ask a question about your documents: ")
    if user_question:
        handle_user_question(user_question)
    
    st.markdown("""
        <style>
        .chat-box {
            max-height: 400px;
            overflow-y: scroll;
            padding: 1rem;
            border: 1px solid #ccc;
            border-radius: 8px;
            background-color: #f9f9f9;
        }
        </style>
    """, unsafe_allow_html=True)

    with chat_container:
        # Display chat history
        display_chat()
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here",accept_multiple_files=True)
        if pdf_docs:
            uploaded_names = [f.name for f in pdf_docs]
            processed_names = [f.name for f in st.session_state.get("processed_files", [])]
            if uploaded_names != processed_names:
                with st.spinner("Processing"):
                    # Get the text
                    raw_text = get_pdf_text(pdf_docs)
                
                    # Get the text chunks
                    chunks=get_text_chunks(raw_text)

                    # Create vector DB
                    vector_DB=get_vector_store(chunks)

                    #create conversation chain
                    st.session_state.conversation=get_conversation_chain(vector_DB)
                    st.session_state.processed_files = pdf_docs
                
                st.success("Tài liệu đã được xử lý thành công ✅")

     
            
if  __name__ == "__main__":
    main()