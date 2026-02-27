from typing import List, TypedDict, Optional

class AgentState(TypedDict):
    """
    Represents the state of our development graph.
    """
    query: str
    intent: Optional[str]
    retrieved_docs: List[str]
    context_valid: bool
    generation: str
    confidence_score: float
    chat_history: List[dict]
    source_documents: List[str]
    error: Optional[str]
