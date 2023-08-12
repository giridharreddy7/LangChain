import streamlit as st
import pandas as pd
import json
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from agent import query_agent, create_agent
from htmlTemplates import css, bot_template, user_template

# Code from app.py
def app():
    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(text_chunks):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        vectorstore.save_local("faiss_index")
        new_persist_db = FAISS.load_local("faiss_index", embeddings)
        return new_persist_db

    def get_persist_vectorstore():
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        new_persist_db = FAISS.load_local("faiss_index", embeddings)
        return new_persist_db

    def get_conversation_chain(vectorstore):
        llm = ChatOpenAI(temperature=0.2)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            verbose=True,
            memory=memory
        )
        return conversation_chain

    def handle_userinput(user_question):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Customer Insights using Vector Search")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
        else:
            vectorstore = get_persist_vectorstore()
        st.session_state.conversation = get_conversation_chain(vectorstore)


# Code from interface.py
def interface():
    def decode_response(response: str) -> dict:
        return json.loads(response)

    def write_response(response_dict: dict):
        if "answer" in response_dict:
            st.write(response_dict["answer"])
        if "bar" in response_dict:
            data = response_dict["bar"]
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.bar_chart(df)
        if "line" in response_dict:
            data = response_dict["line"]
            df = pd.DataFrame(data)
            st.line_chart(df)

    st.title('Conversational Agent')
    st.write('Welcome to the conversational agent. How can I help you today?')
    user_input = st.text_input('Please enter your question:')
    if user_input:
        response = query_agent(user_input)
        response_dict = decode_response(response)
        write_response(response_dict)


def main():
    load_dotenv()
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ("App", "Interface"))
    if choice == "App":
        app()
    elif choice == "Interface":
        interface()

if __name__ == "__main__":
    main()

