
from langgraph.graph import MessagesState


class AICompanionState(MessagesState):
    """State class for the AI Companion workflow.

    Extends MessagesState to track conversation history and maintains the last message received.
    Attributes:
        messages (list): List of messages in the conversation.
        last_message (str): The last message received in the conversation.
        nomor (str): Nomor related to the current context.
        tahun (str): Tahun related to the current context.
        docs (list): List of documents related to the current context.
        output (str): Output from the last executed node.
        memory_context (str): Context for memory management.
        summary (str): Summary of the conversation or context.
    """
    nomor: str = None
    tahun: str = None
    docs: list = None
    output: str = None
    memory_context: str = None
    summary: str = None
    type: str = None
    filters: dict = None
    is_important: bool = False
