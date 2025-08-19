import re

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from ai_companion.settings import settings


def get_chat_model(temperature: float = 0.7):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=temperature,
    )
    # return ChatGroq(
    #     api_key=settings.GROQ_API_KEY,
    #     model_name=settings.TEXT_MODEL_NAME,
    #     temperature=temperature,
    # )


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))
