from langchain_core.messages import AIMessageChunk, HumanMessage
import asyncio
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ai_companion.graph.edges import (
    select_workflow,
    should_summarize_conversation,
)
from ai_companion.graph.nodes import (
    context_injection_node,
    conversation_node,
    memory_extraction_node,
    memory_injection_node,
    router_node,
    summarize_conversation_node,
)
from ai_companion.graph.state import AICompanionState


@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(AICompanionState)

    # Add all nodes
    # graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    # graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    # graph_builder.add_node("memory_injection_node", memory_injection_node)
    # graph_builder.add_node("conversation_node", conversation_node)
    # graph_builder.add_node("image_node", image_node)
    # graph_builder.add_node("audio_node", audio_node)
    # graph_builder.add_node("summarize_conversation_node",
    #                        summarize_conversation_node)

    # graph.add_node("extract_nomor_tahun", extract_nomor_tahun)

    # graph.add_node("retrieve_docs", retrieve_docs)
    # Define the flow
    # First extract memories from user message
    # graph_builder.add_edge(START, "memory_extraction_node")
    graph_builder.add_edge(START, "context_injection_node")

    # # Then determine response type
    # graph_builder.add_edge("memory_extraction_node", "router_node")

    # # Then inject both context and memories
    # graph_builder.add_edge("router_node", "context_injection_node")
    # graph_builder.add_edge("context_injection_node", "memory_injection_node")

    # # Then proceed to appropriate response node
    # graph_builder.add_conditional_edges(
    #     "memory_injection_node", select_workflow)

    # # Check for summarization after any response
    # graph_builder.add_conditional_edges(
    #     "conversation_node", should_summarize_conversation)
    # graph_builder.add_conditional_edges(
    #     "image_node", should_summarize_conversation)
    # graph_builder.add_conditional_edges(
    #     "audio_node", should_summarize_conversation)
    graph_builder.add_edge("context_injection_node", END)

    return graph_builder


# Compiled without a checkpointer. Used for LangGraph Studio
if __name__ == "__main__":
    async def main():
        graph = create_workflow_graph().compile()
        # query = "Undang-Undang apa saja yang membahas sengketa pajak! jawab dengan singkat poin per poin nomor undang-undangnya dan tahunnya!"
        # query = "Undang undang tahun 2015 pasal 1 tentang apa?"
        query = "coba bandingin pasal 3 dan pasal 7 tahun 2020?"
        result = await graph.ainvoke({"messages": [HumanMessage(content=query)]})
        print("✅ Pertanyaan:", query)
        print("\n📜 Jawaban:\n", result)

    asyncio.run(main())
