import sys
from unittest.mock import MagicMock
# Mock transformers/torch to avoid DLL errors
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.state import AgentState
from src.rag_engine import RAGEngine

llm = ChatOpenAI(
    model=(os.getenv("MODEL_NAME") or "google/gemini-2.0-flash-lite-preview-02-05:free").replace("\n", "").replace("\r", "").strip(),
    openai_api_key=(os.getenv("OPENROUTER_API_KEY") or "").strip(),
    openai_api_base=(os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip(),
    temperature=0
)
print(f"DEBUG: Initialized LLM with model: '{llm.model_name}'")
rag = RAGEngine()

def input_processing_node(state: AgentState) -> Dict[str, Any]:
    print("--- INPUT PROCESSING ---")
    query = state["query"].strip()
    return {"query": query}

def intent_routing_node(state: AgentState) -> Dict[str, Any]:
    print("--- INTENT ROUTING ---")
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following user query and determine if it requires information from a knowledge base "
        "or if it is a general greeting/chit-chat.\n\n"
        "Query: {query}\n\n"
        "Return only one word: 'RAG' or 'GENERAL'."
    )
    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"query": state["query"]}).strip().upper()
    return {"intent": intent}

def retrieval_node(state: AgentState) -> Dict[str, Any]:
    print("--- RETRIEVAL ---")
    retriever = rag.get_retriever()
    if not retriever:
        return {"retrieved_docs": [], "error": "No knowledge base found."}
    
    docs = retriever.invoke(state["query"])
    doc_contents = [d.page_content for d in docs]
    return {"retrieved_docs": doc_contents, "source_documents": doc_contents}

def context_validation_node(state: AgentState) -> Dict[str, Any]:
    print("--- CONTEXT VALIDATION ---")
    if not state["retrieved_docs"]:
        return {"context_valid": False}
    
    prompt = ChatPromptTemplate.from_template(
        "Given the following context and a query, determine if the context contains enough information "
        "to accurately answer the query. Respond with 'YES' or 'NO' followed by a confidence score (0.0 to 1.0).\n\n"
        "Context: {context}\n\n"
        "Query: {query}\n\n"
        "Response format: [YES/NO], [SCORE]"
    )
    
    context_str = "\n".join(state["retrieved_docs"])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_str, "query": state["query"]})
    
    valid = "YES" in response.upper()
    try:
        score = float(response.split(",")[-1].strip())
    except:
        score = 0.5
        
    return {"context_valid": valid, "confidence_score": score}

def response_generation_node(state: AgentState) -> Dict[str, Any]:
    print("--- RESPONSE GENERATION ---")
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the user query using ONLY the provided context. "
        "If the context doesn't have the answer, say you don't know.\n\n"
        "Context: {context}\n\n"
        "Query: {query}"
    )
    
    context_str = "\n".join(state["retrieved_docs"])
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_str, "query": state["query"]})
    return {"generation": response}

def fallback_node(state: AgentState) -> Dict[str, Any]:
    print("--- FALLBACK ---")
    if state.get("intent") == "GENERAL":
        prompt = ChatPromptTemplate.from_template(
            "The user is engaging in general conversation. Respond politely and briefly.\n\n"
            "Query: {query}"
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"query": state["query"]})
    else:
        response = "I'm sorry, I couldn't find enough information in my knowledge base to answer your question accurately."
    
    return {"generation": response}

def response_formatter_node(state: AgentState) -> Dict[str, Any]:
    print("--- RESPONSE FORMATTER ---")
    gen = state["generation"]
    # Final cleanup or formatting
    formatted_response = gen.strip()
    return {"generation": formatted_response}
