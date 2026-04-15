from typing import List

from app.adapters.llm.factory import LLMAdapterFactory
from app.config import LLMProvider, settings
from app.schemas.summarize import ChatMessage


def _format_chat_history(chats: List[ChatMessage]) -> str:
    return "\n".join(f"{chat.role}: {chat.content}" for chat in chats)


def _build_messages(chats: List[ChatMessage]) -> list[dict[str, str]]:
    system_prompt = """You are a context compressor specializing in legal and document-related conversations. Your job is to compress the provided chat history into a compact system message that preserves the most useful context for future turns.

    OUTPUT FORMAT:
    Plain text only. No markdown, no bullets, no headers.
    Use short dense lines separated by a single newline.
    Each line should represent a single key fact, issue, decision, or requirement.
    Keep the total output under 25 lines.

    PRIORITY:
    - Highest: Active issues, pending actions, latest decisions
    - Medium: Legal clauses, obligations, parties, governing law
    - Lowest: Background context

    INCLUDE:
    - Document names, types, versions, and purpose
    - Legal clauses, obligations, and key terms
    - Requested changes and their current status
    - Decisions made and reasoning (if relevant)
    - Deadlines, dates, jurisdiction, governing law
    - Open issues or unresolved conflicts

    EXCLUDE:
    - Greetings, acknowledgements, repetition
    - Superseded or outdated information
    - Low-signal chatter

    RULES:
    - Be as concise as possible; fewer lines is better as long as no key fact is lost
    - Each fact must be unique; never restate the same information in different words
    - Keep total output under 20 lines; if forced to drop, drop lowest priority first
    - Prefer the most recent valid information when conflicts exist
    - Preserve exact legal terms and party names
    - Do not alter legal meaning
    - Do not invent facts
    - Never use em dashes
    - If nothing is worth preserving, output exactly: [NO_FACTS]
    """

    chat_history = _format_chat_history(chats)
    user_message = f"""Summarize the chat history below into compact reusable context.

    Chat history:
    {chat_history}

    Return only the compact summary text."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


async def summarize_response(chats: List[ChatMessage], provider: LLMProvider | None = None) -> List[ChatMessage]:
    tail_message_count = settings.tail_message_count

    if len(chats) <= tail_message_count:
        return chats

    summary_source = chats[:-tail_message_count]
    tail_messages = chats[-tail_message_count:]

    adapter = LLMAdapterFactory.create(provider=provider)
    summary = await adapter.chat(_build_messages(summary_source))

    if summary.strip() == "[NO_FACTS]":
        return list(tail_messages)

    return [
        ChatMessage(role="system", content=f"[SUMMARY OF EARLIER CONVERSATION]\n{summary}"),
        *tail_messages,
    ]