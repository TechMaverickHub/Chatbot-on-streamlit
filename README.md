# 📄 PDF Question Answering App using LangChain, FAISS & HuggingFace

This Streamlit app allows you to upload a PDF file and ask questions about its content using a Retrieval-Augmented Generation (RAG) pipeline. It leverages LangChain, FAISS, Hugging Face models, and Streamlit for an interactive interface.

---

## 🚀 Features

- Upload any PDF file.
- Ask questions related to the document.
- Uses `sentence-transformers` for embeddings.
- Retrieves relevant chunks using FAISS.
- Generates answers using Hugging Face models (`bigscience/bloom`).
- Displays source text used for the answer.

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Transformers
- PyPDFLoader
- dotenv

---

## 📦 Installation and run the program

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/pdf-qa-app.git
cd pdf-qa-app

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

streamlit run app2.py


