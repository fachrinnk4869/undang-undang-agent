from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from ai_companion.core.prompts import CHARACTER_CARD_DIRECT_PROMPT, CHARACTER_CARD_LIST_PROMPT, CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from ai_companion.graph.utils.helpers import AsteriskRemovalParser, get_chat_model


class RouterResponse(BaseModel):
    response_type: str = Field(
        description="The response type to give to the user. It must be one of: 'conversation', 'image' or 'audio'"
    )


def get_router_chain():
    model = get_chat_model(
        temperature=0.3).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTER_PROMPT), MessagesPlaceholder(variable_name="messages")]
    )

    return prompt | model


def get_character_response_chain(summary: str = ""):
    model = get_chat_model()
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Ava and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()


def get_list_response(summary: str = ""):
    model = get_chat_model()
    system_message = CHARACTER_CARD_LIST_PROMPT

    if summary:
        system_message += f"\n\nSummary of these list of context take just point per point: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()


def get_direct_response(summary: str = ""):
    model = get_chat_model()
    system_message = CHARACTER_CARD_DIRECT_PROMPT

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    return prompt | model | AsteriskRemovalParser()
