import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
import os
import tempfile

# Function to preprocess a single PDF and create FAISS index
def preprocess_and_create_index(uploaded_file):
    # Save the content of the UploadedFile to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())

    try:
        # Create FAISS index using the temporary file
        loader = PyPDFLoader(temp_file.name)
        pages = loader.load_and_split()
        faiss_index = FAISS.from_documents(pages, HuggingFaceEmbeddings())
        return faiss_index
    finally:
        # Clean up the temporary file
        os.remove(temp_file.name)

def main():
    st.title("Question Answering with Document")

    # Upload PDF files
    #uploaded_files = st.file_uploader("Upload one or more PDF documents", type=["pdf"], accept_multiple_files=True)
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        # Call the preprocessing function
        with st.spinner("Uploading and Processing..."):
        # Process the file, e.g., preprocess_and_create_index
            faiss_index = preprocess_and_create_index(uploaded_file)
        st.success("File processed successfully!")


            # Use FAISS index as a retriever
        retriever = faiss_index.as_retriever()

            # Question-answering pipeline
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        model = ChatOpenAI(
                model_name='TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ',
                openai_api_base='http://20.124.240.6:8083/v1',
                openai_api_key="EMPTY",
                streaming=False,
            )

        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

            # Ask a question and display the answer
        question = st.text_input("Ask a question:")
        get_answer_button = st.button("Get Answer", key="get_answer_button")

        if get_answer_button:
            with st.spinner("Processing..."):
                answer = chain.invoke(question)
            st.success(f"Answer: {answer}")


if __name__ == "__main__":
    main()
