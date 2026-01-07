import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API Keys
load_dotenv()

# 1. Setup the Embedding Model (Must match the Loader!)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. Connect to Pinecone
index_name = "lol-nexus"
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
# We retrieve the top 3 most relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Setup Groq (The High-Speed LLM)
# We use Llama3-70b for high intelligence but extreme speed
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# 4. Define the Prompt
# This system prompt prevents hallucinations by forcing it to use context
template = """
You are a League of Legends Analyst for Patch 14.1.
Answer the user's question strictly based on the context provided below.
If the context does not contain the answer, say "I couldn't find that in the Patch 14.1 notes."

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. Build the Chain
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def get_response(query: str):
    """
    Runs the RAG chain and measures latency.
    """
    start_time = time.time()
    
    try:
        response = rag_chain.invoke(query)
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "latency_ms": 0}
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    return {
        "answer": response,
        "latency_ms": round(latency_ms, 2)
    }

# Simple test to run this file directly
if __name__ == "__main__":
    # Test Question
    q = "What buffs did Hwei get?"
    print(f"Testing Query: {q}")
    result = get_response(q)
    print(f"Answer: {result['answer']}")
    print(f"Time: {result['latency_ms']}ms")