# ready_tensor_bot.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Custom Prompt Template for Research Papers
research_prompt_template = """
You are an expert assistant helping readers understand technical publications from Ready Tensor.
Use the following context from the publication(s) to answer the question clearly and concisely.

If the information is not present, say "I couldn't find that information in the available publications."

Be precise when discussing methods, models, tools, assumptions, or limitations.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=research_prompt_template, input_variables=["context", "question"])

def create_qa_chain():
    # Load embeddings (must match ingestion)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load vector DB
    db = FAISS.load_local("faiss_ready_tensor", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Initialize Gemini
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
        convert_system_message_to_human=True
    )

    # Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa

def chat():
    print("🧠 Ready Tensor Publication Explorer")
    print("🔍 Ask questions about any publication (e.g., 'What models were used?', 'Limitations?', 'Summary?')")
    print("Type 'quit' to exit.\n")

    qa_chain = create_qa_chain()

    while True:
        question = input("💬 Question: ").strip()
        if question.lower() == "quit":
            print("👋 Goodbye! Thanks for exploring Ready Tensor.")
            break
        if not question:
            continue

        response = qa_chain.invoke({"query": question})
        answer = response["result"].strip()

        print(f"\n🟢 Answer: {answer}")

        # Show sources
        print("\n📎 Source(s):")
        seen = set()
        for doc in response["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            if source not in seen:
                print(f"  - From: {source}")
                seen.add(source)
        print("-" * 60)

if __name__ == "__main__":
    chat()