from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.nodes import (
    input_processing_node,
    intent_routing_node,
    retrieval_node,
    context_validation_node,
    response_generation_node,
    fallback_node,
    response_formatter_node
)

def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("process_input", input_processing_node)
    workflow.add_node("route_intent", intent_routing_node)
    workflow.add_node("retrieve", retrieval_node)
    workflow.add_node("validate_context", context_validation_node)
    workflow.add_node("generate", response_generation_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("format_response", response_formatter_node)

    # Set Entry Point
    workflow.set_entry_point("process_input")

    # Define Edges
    workflow.add_edge("process_input", "route_intent")

    # Conditional Edges from Intent Routing
    def decide_retrieval_path(state):
        if state["intent"] == "RAG":
            return "retrieve"
        return "fallback"

    workflow.add_conditional_edges(
        "route_intent",
        decide_retrieval_path,
        {
            "retrieve": "retrieve",
            "fallback": "fallback"
        }
    )

    workflow.add_edge("retrieve", "validate_context")

    # Conditional Edges from Validation
    def decide_generation_path(state):
        if state["context_valid"]:
            return "generate"
        return "fallback"

    workflow.add_conditional_edges(
        "validate_context",
        decide_generation_path,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )

    workflow.add_edge("generate", "format_response")
    workflow.add_edge("fallback", "format_response")
    workflow.add_edge("format_response", END)

    return workflow.compile()

# Initialize the compiled graph
app_graph = create_graph()
