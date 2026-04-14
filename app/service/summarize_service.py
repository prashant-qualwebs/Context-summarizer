from app.adapters.llm.factory import LLMAdapterFactory


def _build_messages(assistant_response: str) -> list[dict[str, str]]:
    system_prompt = """You are a context compressor. Your job is to extract and compress all persistently useful facts from an assistant response into a reusable context snippet.

    OUTPUT FORMAT:
    Plain text only. No markdown, no bullets, no headers.
    Short dense phrases. Full sentences only when precision requires it.
    Do not pad. Output only what exists.

    HARD RULE: Never use em dashes. Use a colon to separate term names from facts.
    Correct: "Term 1 Termination: 24-hour notice; no cure period."
    Incorrect: "Term 1 — Termination: 24-hour notice."

    EXTRACT IN PRIORITY ORDER:

    LEGAL FACTS (if present):
    1. Obligations, liabilities, rights, restrictions
    2. Deadlines, dates, triggers, timelines
    3. Monetary terms: fees, damages, penalties, interest rates
    4. Claims, defenses, disputes, unresolved issues, risks
    5. Clauses, contract sections, document names, statutes cited
    6. Parties, roles, relationships
    7. Jurisdiction, governing law, courts
    8. Legal conclusions, rulings, findings, AI assessments (e.g. standard, problematic, no playbook issue, flagged for review). Preserve exactly.
    9. Verbatim clause text, defined terms, exact amounts. Preserve word for word.
    10. User intent: what the user is trying to accomplish

    NON-LEGAL FACTS (extract these when legal facts are absent or minimal):
    - Quantitative results: counts, scores, metrics, calculations, comparisons
    - Factual findings: what was found, measured, rated, or concluded
    - Named entities: people, companies, products, documents referenced
    - User intent and task context: what the user asked, what was analyzed
    - Any output that would be useful to recall in a follow-up prompt

    COMPRESSION RULES:
    - Compress aggressively. Every word must carry information.
    - Merge related facts into one line
    - No repetition
    - Preserve exact wording for amounts, names, defined terms, and conclusions
    - For exclusions and carve-outs, preserve all qualifying conditions
    - If conflicting facts exist, include both and mark: [CONFLICT]
    - Do NOT add interpretation or infer facts not present
    - Strip all explanations, caveats, hedging, examples, analogies, boilerplate
    - Retain conclusions only, not reasoning

    STRIP ALWAYS:
    Recitals, whereas clauses, execution blocks, signature lines, standard disclaimer language,
    boilerplate indemnification recitals, hedging phrases (e.g. it is important to note,
    generally speaking, you may want to consult), and any content fully re-inferable from context.

    FALLBACK RULE:
    There is no "nothing to extract" scenario. Every assistant response contains something worth compressing.
    If legal facts are absent, extract non-legal facts. Always return something useful.
    """

    user_message = f"""Extract a persistent context snippet from the assistant response below.
    This snippet will be stored as memory and injected into future prompts.

    Assistant response:
    {assistant_response}

    Return the compressed context snippet. Plain text only."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


async def summarize_response(assistant_response: str, provider: str = None) -> str:
    messages = _build_messages(assistant_response=assistant_response)

    adapter = LLMAdapterFactory.create(provider=provider)
    return await adapter.chat(messages)