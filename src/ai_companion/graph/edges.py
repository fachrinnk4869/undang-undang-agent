from langgraph.graph import END
from typing_extensions import Literal
from langchain_core.messages import AIMessage
from ai_companion.graph.state import AICompanionState
from ai_companion.settings import settings


def should_summarize_conversation(
    state: AICompanionState,
) -> Literal["summarize_conversation_node", "__end__"]:
    messages = state["messages"]

    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return "summarize_conversation_node"

    return END


def should_summarize_continue(state: AICompanionState) -> bool:
    # kalau sudah selesai, sebelum END, copy dulu summary ke messages
    if state["current_index"] >= len(state["chunks"]):
        if "summary" in state:
            # tambahin ke messages biar kebawa
            state["messages"].append(AIMessage(content=state["summary"]))
        return False
    return True


def select_workflow(
    state: AICompanionState,
) -> Literal["list_node", "direct_node", "comparison_node", "summary_node", "count_node", "exists_node", "mixed_node"]:
    """
    Select the appropriate workflow based on the type of request in the state.
    The type can be one of the following:
    - "list": The user asks for all results that match a filter
    - "direct": The user asks for a specific detail or explanation from a single best match
    - "comparison": The user requests a comparison between two or more results
    - "summary": The user asks for a summary of multiple results
    - "count": The user asks for the number or statistics of matching results
    - "exists": The user asks whether certain data exists
    - "mixed": The user combines multiple request types in one message
    """

    typed = state["type"]

    if typed == "list":
        return "list_node"
    elif typed == "direct":
        return "direct_node"
    elif typed == "comparison":
        return "comparison_node"
    elif typed == "summary":
        return "summary_node"
    elif typed == "count":
        return "count_node"
    elif typed == "exists":
        return "exists_node"
    elif typed == "mixed":
        return "mixed_node"
    else:
        raise ValueError(f"Unknown type: {typed}")
