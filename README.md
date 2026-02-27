# AskGalore RAG Chatbot Avatar

A premium RAG-based chatbot built with **LangGraph** and **Streamlit**.

## Features
- **Modular Workflow**: Orchestrated by LangGraph with specialized nodes.
- **Dynamic Routing**: Automatically detects intent (RAG vs. General).
- **Hallucination Safety**: Context validation node ensures answers are grounded.
- **Premium UI**: Sleek dark-themed Streamlit interface.
- **Knowledge Base**: Support for PDF and TXT document indexing.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file (one has been initialized for you) and add your OpenAI API Key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Run the App**:
   ```bash
   streamlit run app.py
   ```

## Workflow Nodes
- **Input Processing**: Sanitizes the user query.
- **Intent Routing**: Determines if RAG retrieval is necessary.
- **Retrieval**: Fetches relevant context from FAISS.
- **Context Validation**: Verifies retrieval quality to prevent hallucinations.
- **Response Generation**: LLM generates the answer using validated context.
- **Fallback**: Handles cases with low confidence or general chat.
- **Response Formatter**: Final polish of the output.

## Sample Data
Sample documents about AskGalore, LangGraph, and RAG have been provided in the `data/` directory. Use the "Index Documents" button in the sidebar to load them.
