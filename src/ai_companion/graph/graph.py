from langchain_core.messages import AIMessageChunk, HumanMessage
import asyncio
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ai_companion.graph.edges import (
    select_workflow,
    should_summarize_continue,
    should_summarize_conversation,
)
from ai_companion.graph.nodes import (
    context_injection_node,
    list_node,
    direct_node,
    comparison_node,
    summarize_node,
    summary_node,
    count_node,
    exists_node,
    mixed_node,
)
from ai_companion.graph.state import AICompanionState


@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(AICompanionState)

    # Add all nodes
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("list_node", list_node)
    graph_builder.add_node("direct_node", direct_node)
    graph_builder.add_node("comparison_node", comparison_node)
    graph_builder.add_node("summary_node", summary_node)
    graph_builder.add_node("count_node", count_node)
    graph_builder.add_node("exists_node", exists_node)
    graph_builder.add_node("mixed_node", mixed_node)
    graph_builder.add_node("summarize_node", summarize_node)
    # rerank
    # Define the flow
    graph_builder.add_edge(START, "context_injection_node")

    graph_builder.add_conditional_edges(
        "context_injection_node", select_workflow)

    graph_builder.add_edge("direct_node", "summarize_node")

    graph_builder.add_conditional_edges(
        "summarize_node",
        should_summarize_continue,
        {
            True: "summarize_node",  # loop lagi
            False: END          # selesai
        }
    )

    graph_builder.add_edge("list_node", END)
    graph_builder.add_edge("comparison_node", END)
    graph_builder.add_edge("summarize_node", END)
    graph_builder.add_edge("count_node", END)
    graph_builder.add_edge("exists_node", END)
    graph_builder.add_edge("mixed_node", END)

    return graph_builder


# Compiled without a checkpointer. Used for LangGraph Studio
if __name__ == "__main__":
    async def main():
        graph = create_workflow_graph().compile()
        # query = "Undang-Undang apa saja yang membahas sengketa pajak! jawab dengan singkat poin per poin nomor undang-undangnya dan tahunnya!"
        query = "Undang undang tahun 2015 pasal 1 tentang apa?"
        # query = "coba sebutkan apa saja undang undang yang membahas pajak, sebutkan nomor undang-undangnya dan tahunnya!"
        result = await graph.ainvoke({"messages": [HumanMessage(content=query)]}, {"recursion_limit": 100})
        print("✅ Pertanyaan:", query)
        print("\n📜 Jawaban:\n", result['messages'][-1])

    asyncio.run(main())
