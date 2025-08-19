# from sentence_transformers import CrossEncoder
import os
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from ai_companion.graph.state import AICompanionState
from ai_companion.graph.utils.chains import (
    get_character_response_chain,
    get_direct_response,
    get_list_response,
    get_router_chain,
)
from ai_companion.graph.utils.helpers import (
    get_chat_model
)
from ai_companion.memory.memory_manager import get_memory_manager, get_context_manager
from ai_companion.settings import settings


async def router_node(state: AICompanionState):
    chain = get_router_chain()
    response = await chain.ainvoke({"messages": state["messages"][-settings.ROUTER_MESSAGES_TO_ANALYZE:]})
    return {"workflow": response.response_type}


async def memory_extraction_node(state: AICompanionState):
    """Extract and store important information from the last message."""
    if not state["messages"]:
        return {}

    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memories(state["messages"][-1])
    return {}

memory_context_manager = get_context_manager()


async def context_injection_node(state: AICompanionState):
    """Extract and store related information about undang undang from the last message."""
    if not state["messages"]:
        return {}

    result = await memory_context_manager.extract_context(state["messages"][-1])
    print(f"Extracted context: {result}")
    return {
        "type": result['type'],
        "filters": result['filters'],
        "is_important": result['is_important']
    }


async def list_node(state: AICompanionState):
    """Handle the 'list' workflow."""
    # Implement the logic for listing results based on filters
    filters = state.get("filters", {})
    query = state["messages"][-1].content

    contexts = memory_context_manager.vector_store.search_memories_by_filters(query,
                                                                              filters)
    memory_context = "\n".join(str(c) for c in contexts)
    # print(f"Memory context for filters {filters}: {memory_context}")

    chain = get_list_response(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "memory_context": contexts,
        },
        # config,
    )
    return {"messages": AIMessage(content=response)}

# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


async def direct_node(state: AICompanionState):
    """Handle the 'list' workflow."""
    # Implement the logic for listing results based on filters
    # Implement the logic for listing results based on filters
    filters = state.get("filters", {})

    chunks = memory_context_manager.vector_store.search_memories_just_by_filters(
        filters)

    # chain = get_direct_response()

    return {
        # "chain": chain,
        "chunks": chunks
    }


async def llm_node(state: AICompanionState):
    """Handle llm node."""
    response = await state["chain"].ainvoke(
        {
            "messages": state["messages"],
            "memory_context": state['memory_context'],
        },
        # config,
    )
    return {"messages": AIMessage(content=response)}


async def summarize_node(state: AICompanionState):
    """Summarize the conversation."""
    model = get_chat_model()
    summary_so_far = state.get("summary", "")
    idx = state.get("current_index", 0)
    chunk = state['chunks'][idx:idx + 10] if idx + \
        10 < len(state['chunks']) else state['chunks'][idx:]

    if summary_so_far:
        prompt = f"""
        Ringkasan sementara:
        {summary_so_far}

        Tambahan teks:
        {str(chunk)}

        Tolong perbarui ringkasan dengan hanya menuliskan
        poin-poin inti (maksimal 3-5 poin), tanpa detail tambahan. gunakan bahasa yang baik dan benar, jangan dipaksakan hanya satu paragraf kalau memang membutuhkan paragraph lebih dari satu.
        """
    else:
        prompt = f"""
        Ringkas isi teks berikut menjadi poin-poin utama saja
        (maksimal 3-5 poin, kalimat singkat):

        {chunk}
        """

    new_summary = await model.ainvoke(prompt)
    print(f"New summary: {new_summary.content}")
    return {
        "summary": new_summary.content,
        "current_index": idx + 10
    }


async def summary_node(state: AICompanionState):
    """Handle the 'list' workflow."""
    # Implement the logic for listing results based on filters
    filters = state.get("filters", {})

    memory_context = memory_manager.vector_store.search_memories_by_filters(
        filters)

    chain = get_list_response(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "memory_context": memory_context,
        },
        # config,
    )
    return {"messages": AIMessage(content=response)}


async def mixed_node(state: AICompanionState):
    """Handle the 'list' workflow."""
    # Implement the logic for listing results based on filters
    filters = state.get("filters", {})

    memory_context = memory_manager.vector_store.search_memories_by_filters(
        filters)

    chain = get_list_response(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "memory_context": memory_context,
        },
        # config,
    )
    return {"messages": AIMessage(content=response)}


async def comparison_node(state: AICompanionState):
    """Handle the 'list' workflow."""
    # Implement the logic for listing results based on filters
    filters = state.get("filters", {})

    memory_context = memory_manager.vector_store.search_memories_by_filters(
        filters)

    chain = get_list_response(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "memory_context": memory_context,
        },
        # config,
    )
    return {"messages": AIMessage(content=response)}


async def count_node(state: AICompanionState):
    """Handle the 'list' workflow."""
    # Implement the logic for listing results based on filters
    filters = state.get("filters", {})

    memory_context = memory_manager.vector_store.search_memories_by_filters(
        filters)

    chain = get_list_response(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "memory_context": memory_context,
        },
        # config,
    )
    return {"messages": AIMessage(content=response)}


async def exists_node(state: AICompanionState):
    """Handle the 'list' workflow."""
    # Implement the logic for listing results based on filters
    filters = state.get("filters", {})

    memory_context = memory_manager.vector_store.search_memories_by_filters(
        filters)

    chain = get_list_response(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "memory_context": memory_context,
        },
        # config,
    )
    return {"messages": AIMessage(content=response)}


async def conversation_node(state: AICompanionState, config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_character_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    return {"messages": AIMessage(content=response)}


def memory_injection_node(state: AICompanionState):
    """Retrieve and inject relevant memories into the character card."""
    memory_manager = get_memory_manager()

    # Get relevant memories based on recent conversation
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)

    # Format memories for the character card
    memory_context = memory_manager.format_memories_for_prompt(memories)

    return {"memory_context": memory_context}


def extract_nomor_tahun(state: AICompanionState):
    question = state["input"]
    prompt = (
        f"Ambil nomor dan tahun undang undang dari pertanyaan berikut dengan akurat:\n"
        f"{question}\n\n"
        f"Format jawaban: nomor=[nomor], tahun=[tahun]"
    )
    if model_style == "gemini":
        response = llm.invoke(prompt).content
    else:
        response = llm.invoke(prompt)
    match = re.search(r"nomor\s*=\s*(\d+).+tahun\s*=\s*(\d+)",
                      response, re.IGNORECASE | re.DOTALL)
    if match:
        nomor, tahun = match.groups()
    else:
        nomor, tahun = None, None
    return {"nomor": nomor, "tahun": tahun}


def retrieve_docs(state):
    nomor, tahun = state.get("nomor"), state.get("tahun")
    # filters = {"nomor": nomor, "tahun": tahun} if nomor and tahun else None
    filters = {
        "$and": [
            {"nomor": {"$eq": nomor}},
            {"tahun": {"$eq": tahun}}
        ]
    } if nomor and tahun else None
    print(f"Retrieving documents with filters: {filters}")
    if filters is None:
        results = vectorstore.similarity_search(
            state["input"], k=10)
    else:
        results = vectorstore.similarity_search(
            state["input"], filter=filters)
    print(f"Retrieving documents found: {results}")
    return {"docs": results}


async def summarize_conversation_node(state: AICompanionState):
    model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Ava and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Ava and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Ava and the user:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [RemoveMessage(
        id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages}
