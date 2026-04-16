from typing import List

from app.adapters.llm.factory import LLMAdapterFactory
from app.config import LLMProvider, settings
from app.schemas.summarize import ChatMessage


def _format_chat_history(chats: List[ChatMessage]) -> str:
    return "\n".join(f"{chat.role}: {chat.content}" for chat in chats)


def _build_messages(chats: List[ChatMessage]) -> list[dict[str, str]]:
    system_prompt = """You are a legal conversation memory compressor. You receive a chat history between a user and an AI assistant working on legal documents (NDAs, contracts, agreements). Your job is to produce a dense context block that the next AI turn can rely on without reading the original history.

    OUTPUT RULES:
    - Plain text only. No markdown, bullets, headers, or em dashes.
    - One fact per line.
    - Hard cap: 30 lines. If you exceed this, drop lowest-priority lines first (see priority order below).
    - Never restate the same fact twice, even in different words.
    - Never invent facts. Never alter legal meaning.
    - If nothing is worth preserving, output exactly: [NO_FACTS]

    ENTITY CONSISTENCY:
    - On the first line, output a name map if any aliases exist.
    Format: ENTITIES: [canonical name] = [alias1, alias2, ...]
    Example: ENTITIES: Acme Corp = "the Company", "Party A", "Client"
    - Use only the canonical name in all subsequent lines.
    - If no aliases exist, skip this line entirely.

    TASK STATE TAGGING:
    - Every task or edit request must be tagged with its state.
    [OPEN]     = requested but not yet addressed
    [DONE]     = completed and accepted
    [REJECTED] = proposed but declined, include brief reason if given
    [PARTIAL]  = started but not fully resolved, include what remains
    - Never omit the tag. If state is ambiguous, tag as [OPEN].
    - Example: [PARTIAL] Clause 4.2 non-compete: narrowed to 12 months, duration accepted, geography scope still unresolved

    COMPRESSION PRIORITY (highest to lowest):
    1. All [OPEN] and [PARTIAL] tasks
    2. [REJECTED] decisions with reasoning
    3. Legal terms: clause names, obligations, restrictions, carve-outs, governing law
    4. Party names, roles, jurisdiction, effective dates, deadlines
    5. [DONE] tasks (keep only if they affect current document state)
    6. Document type, version, purpose

    WHAT TO INCLUDE:
    - Every unresolved or in-progress edit request
    - Accepted and rejected changes with brief reason if given
    - Specific clause references (e.g. "Clause 4.2 non-compete: user wants to narrow to 12 months")
    - Party names exactly as written in the document
    - Any legal terms the user introduced or contested
    - Dates, deadlines, and governing law

    WHAT TO EXCLUDE:
    - Greetings, filler, acknowledgements
    - Anything superseded by a later message
    - Repetition of context already captured in another line
    - [DONE] tasks that have no lasting effect on document state

    CONFLICT RULE:
    If two messages contradict each other, keep only the most recent valid instruction and mark the older one as superseded only if it affects an [OPEN] or [PARTIAL] task.
    """

    user_message = """Compress the chat history below into a reusable legal context block.

    Before compressing:
    1. Identify all entity aliases and output the ENTITIES line if needed.
    2. Determine the state of every task or edit request.
    3. Then compress, ordered by priority.

    Chat history:
    {chat_history}

    Output only the compressed context. No preamble. No explanation."""

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