import tempfile
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


# Load your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Upload PDF
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Input question
query = st.text_input("Ask a question about the uploaded PDF:")

if pdf_file and query:
     with st.spinner("Processing..."):
          
          # ✅ Write uploaded file to a temporary file
          with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name

          # ✅ Now use the file path in PyPDFLoader
          loader = PyPDFLoader(tmp_path)
          documents = loader.load()

          text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
          docs = text_splitter.split_documents(documents)

          # Embedding and FAISS vectorstore
          embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
          db = FAISS.from_documents(docs, embeddings)
          
          retriever = db.as_retriever()

          # Optional: Custom prompt
          prompt_template = PromptTemplate.from_template(
            "Context:\n{context}\n\nQuestion:\n{question}"
          )

          llm = HuggingFaceHub(
            repo_id="bigscience/bloom",
            model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
          )


          # QA Chain
          qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
           )

          response = qa_chain.invoke({"query": query})
          answer = response["result"]
          sources = response["source_documents"]

          st.write("Answer:", answer)

          # Optional: Show sources
          with st.expander("Show source documents"):
            for doc in sources:
                st.markdown(f"**Page Content:**\n{doc.page_content}")