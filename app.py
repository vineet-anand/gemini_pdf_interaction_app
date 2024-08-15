#### Importing Libraries

import os
from dotenv import load_dotenv

import streamlit as st

from io import BytesIO
from PyPDF2 import PdfReader

import google.generativeai as genai

from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


#### API KEY Configuration

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

#### Function to load all PDF Files

def get_pdf_info(pdf_docs):

    text = ""

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

#### Function to split text into chunks

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap= 1000)
    chunks = text_splitter.split_text(text)
    
    return chunks

#### Function to convert chunks into Vectors for Vector Embedding

def get_vector(text_chunks):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_data = FAISS.from_texts(text_chunks, embedding= embeddings)
    vector_data.save_local("faiss_index")

#### Function to get the chain of conversation based on Prompts

def get_conversation_chain():

    prompt_template = """
    Please provide a comprehensive and precise answer based on the context provided. 
    Ensure that every detail is addressed. If the context does not contain the necessary information, 
    respond with "The answer is not available in the context." 
    Refrain from making assumptions or providing incorrect answers.\n\n

    Context:\n
    {context}\n

    Question:\n
    {question}\n

    Answer:

"""
    model = ChatGoogleGenerativeAI(model= "gemini-pro", temperature= 0.6)

    prompt = PromptTemplate(template=prompt_template, input_variables= ['context', 'question'])

    chain = load_qa_chain(model, chain_type= "stuff", prompt=prompt)
    
    return chain


#### Function to get user input and utilizing FAISS for similarity search

def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question)

    # Calling get_conversational_chain to get the output from PDF using prompt templates
    chain = get_conversation_chain()
    response = chain({"input_documents":docs, "question": user_question}, 
                     return_only_outputs=True)
    
    print(response)

    st.write("Reply: ", response["output_text"])
    

#### Main Function to call every relevant function as per the requirements


def main():

    st.set_page_config("Information on Multiple PDF")
    st.header("Find information on PDF with Gemini AI")

    user_question = st.text_input("Ask Questions from the PDF Files")

    # Calling user_input function to accomplish Vector Embedding from all PDF.
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title('Menu:')
        pdf_docs= st.file_uploader("Upload your PDF Files then click Submit & Proceed", accept_multiple_files=True)

        if st.button("Submit & Proceed"):
            with st.spinner("Processing...."):

                # Calling get_pdf_info function to read pdf files and extrack all text from all pages of every PDF Files.
                raw_text = get_pdf_info(pdf_docs)

                # Calling get_text_chunks function to split all raw_text using Recursive Character Text Splitter Function in chunks 
                text_chunks = get_text_chunks(raw_text)

                # Calling get_vector function to convert chunks into Vectors that will do Vector embedding 
                # i.e., Vector Embedding will find out the Similarity in text at the higer dimentional Space by either classifying or Clustering all the vetors.
                get_vector(text_chunks)

                st.success("Done")


if __name__ == "__main__":
    main()