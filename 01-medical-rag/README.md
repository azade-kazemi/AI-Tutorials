## Medical RAG Assistant (Project 01)

This project is part of the **AI-Tutorials** repository. It is a Retrieval-Augmented Generation (RAG) system designed to answer medical questions based **strictly** on the provided PDF documents.

## What does this project do?
Instead of relying on the LLM's general knowledge (which can cause hallucinations in the medical field), this system extracts text from your PDF, breaks it into chunks, stores it in a vector database, and retrieves the most relevant chunks to answer the user's question. The LLM is prompted to cite its sources and say "I don't know" if the answer is not in the document.

## Tech Stack
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L12-v2`)
- **LLM Engine:** Groq API (`openai/gpt-oss-20b` - Fast Inference)
- **PDF Parsing:** PyPDF

## Set up your API key:
Add your Groq API Key: GROQ_API_KEY=your_key_here

## Place your medical PDF file in the folder and update the filename in rag.py:

docs = read_pdf_file('your_file.pdf')
Run the script:

python rag.py
