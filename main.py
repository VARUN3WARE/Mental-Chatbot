import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

def initialize_llm():
    groq_api_key = st.secrets["GROQ_API_KEY"]
    
    if not groq_api_key:
        raise ValueError("API key is missing. Please set the GROQ_API_KEY in the secrets file.")
    
    llm = ChatGroq(
        temperature=0,
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )
    return llm


def create_vector_db():
    loader = DirectoryLoader("data/", glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_db = Chroma.from_documents(texts, embeddings, persist_directory='./chroma_db')
    vector_db.persist()

    st.write("ChromaDB created and data saved")

    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt_templates = """ You are a compassionate mental health chatbot. Respond thoughtfully to the following question:
    {context}
    User: {question}
    Chatbot: """
    PROMPT = PromptTemplate(template=prompt_templates, input_variables=['context', 'question'])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

def main():
    st.title("Chatbot for Mental Health Support")
    st.write("This chatbot responds thoughtfully to your mental health queries.")

    llm = initialize_llm()

    db_path = "/content/chroma_db"

    if not os.path.exists(db_path):
        vector_db = create_vector_db()
    else:
        embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    qa_chain = setup_qa_chain(vector_db, llm)

    # Input for user query
    user_input = st.text_input("Ask a question", "")

    if user_input:
        response = qa_chain.run(user_input)
        st.write(f"Chatbot: {response}")

    # Exit functionality
    if st.button("Exit"):
        st.write("Chatbot: Take Care of yourself, Goodbye!")

if __name__ == "__main__":
    main()
