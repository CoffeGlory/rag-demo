# RAG Demo App

A simple end-to-end **Retrieval-Augmented Generation (RAG)** demo built with:

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Document parsing**: PyPDF
- **(Planned)** Embedding + Vector DB (Pinecone)

This project demonstrates a full RAG pipeline:
PDF upload → text chunking → retrieval → answer generation.

# How to Run
Option A: 
- clone repo
- install requirement
- set environment variables
- run start.bat or manually run app.py and /backend/main

Option B:
- Backend deployed on Render: https://nasm-rag.onrender.com/
- Frontend deployed on Render: https://nasm-rag-frontend.onrender.com/
- Both deployed may requires account on render and needs to cold start for minutes.

# Future Improvement
- Better document management
- Let user upload pdf and use it locally without messing the general Vector DB
- Better chunking strategies to give accurate on identifying question range.
