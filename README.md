# Ready Tensor Publication Explorer

A RAG-powered chatbot that answers questions about Ready Tensor publications using:
- LangChain
- Google Gemini 
- FAISS + Hugging Face embeddings

## Features
- Ask natural language questions
- Get answers with source attribution
- Supports PDF, DOCX, HTML, and text formats

## Setup
1. `pip install -r requirements.txt`
2. Get a [Gemini API key](https://aistudio.googlecom/)
3. Save it in `.env`
4. Run `python ingest.py` then `python ready_tensor_bot.py`

> 💡 Data files (PDFs) are not included due to size. Add them to `data/` folder manually.