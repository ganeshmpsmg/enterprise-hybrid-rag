"""
Prompt Builder - Constructs LLM prompts for RAG answer generation.
Handles context formatting, system prompts, and prompt templates.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PromptConfig:
    """Configuration for prompt construction."""

    system_prompt: str
    max_context_length: int = 3000
    include_sources: bool = True
    include_metadata: bool = False
    language: str = "en"


class PromptBuilder:
    """
    Builds structured prompts for the RAG answer generation pipeline.

    RAG Prompt Structure:
    ┌─────────────────────────────────────────┐
    │ SYSTEM: Role, instructions, constraints  │
    ├─────────────────────────────────────────┤
    │ CONTEXT: Retrieved passages (numbered)   │
    ├─────────────────────────────────────────┤
    │ USER: Original question                  │
    └─────────────────────────────────────────┘

    Best practices:
    - Put context BEFORE the question (shown to improve faithfulness)
    - Number sources for citation tracking
    - Instruct model to say "I don't know" if context is insufficient
    - Include source metadata for provenance
    """

    DEFAULT_SYSTEM_PROMPT = """You are an expert AI assistant specialized in Machine Learning and AI research.
Your role is to answer questions accurately using ONLY the provided context passages.

Guidelines:
- Answer based strictly on the provided context. Do not use outside knowledge.
- If the context does not contain enough information to answer the question, say:
  "The provided documents do not contain sufficient information to answer this question."
- Cite sources by referencing [Source N] where N is the passage number.
- Be concise but comprehensive. Prefer bullet points for lists of concepts.
- For technical concepts, provide clear explanations with examples when the context allows.
- Never hallucinate facts, formulas, or references not present in the context."""

    CONTEXT_TEMPLATE = """CONTEXT PASSAGES:
{context}

---
QUESTION: {question}

ANSWER:"""

    def __init__(self, config: Optional[PromptConfig] = None):
        self.config = config or PromptConfig(system_prompt=self.DEFAULT_SYSTEM_PROMPT)

    def build(
        self,
        query: str,
        context_chunks: list[dict],
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Build a complete prompt for RAG answer generation.

        Args:
            query: User's question
            context_chunks: Retrieved and reranked chunks with 'content' and 'metadata'
            system_prompt: Optional override for system prompt

        Returns:
            dict with 'system', 'user' keys (OpenAI message format)
        """
        system = system_prompt or self.config.system_prompt
        context_str = self._format_context(context_chunks)
        user_message = self.CONTEXT_TEMPLATE.format(
            context=context_str,
            question=query,
        )
        return {
            "system": system,
            "user": user_message,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
        }

    def build_conversational(
        self,
        query: str,
        context_chunks: list[dict],
        conversation_history: Optional[list[dict]] = None,
    ) -> list[dict]:
        """
        Build a conversational prompt with history.

        Args:
            query: Current user question
            context_chunks: Retrieved context
            conversation_history: Previous [{"role": "user/assistant", "content": ...}]

        Returns:
            List of message dicts in OpenAI format
        """
        messages = [{"role": "system", "content": self.config.system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Last 3 turns (6 messages)

        # Add current context + question
        context_str = self._format_context(context_chunks)
        user_content = self.CONTEXT_TEMPLATE.format(context=context_str, question=query)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _format_context(self, chunks: list[dict]) -> str:
        """Format retrieved chunks as numbered context passages."""
        if not chunks:
            return "No context available."

        passages = []
        total_chars = 0

        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})

            # Build source line
            source_parts = []
            if metadata.get("file_name"):
                source_parts.append(metadata["file_name"])
            if metadata.get("page_number"):
                source_parts.append(f"p.{metadata['page_number']}")
            if metadata.get("title"):
                source_parts.append(metadata["title"])
            source_line = f"Source: {' | '.join(source_parts)}" if source_parts else ""

            passage = f"[Source {i}] {source_line}\n{content}"

            # Respect max context length
            if total_chars + len(passage) > self.config.max_context_length and i > 1:
                logger.debug(
                    f"Context truncated at {i-1} passages ({total_chars} chars)"
                )
                break

            passages.append(passage)
            total_chars += len(passage)

        return "\n\n".join(passages)

    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (1 token ≈ 4 chars for English)."""
        return len(text) // 4
