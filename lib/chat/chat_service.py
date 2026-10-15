from typing import Any

import litellm
from langfuse import Langfuse
from openai import OpenAI

from lib.config.settings import Settings, get_settings
from lib.db.chat_repository import (
    Interaction,
    get_interaction_history,
    get_session_vector_store_id,
    store_interaction,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using the supplied "
    "website context when it is relevant. If the context does not contain the "
    "answer, say so instead of inventing facts."
)


def _extract_context(vector_store_id: str, query: str, client: OpenAI) -> str:
    search_results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        max_num_results=5,
    )
    excerpts: list[str] = []
    for result in search_results.data:
        for item in result.content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                excerpts.append(text.strip())
    return "\n\n".join(excerpts)


def _build_messages(
    query: str,
    history: list[Interaction],
    retrieved_context: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for interaction in history:
        messages.append({"role": "user", "content": interaction["user"]})
        messages.append({"role": "assistant", "content": interaction["system"]})

    if retrieved_context:
        current_message = (
            f"Relevant website context:\n{retrieved_context}\n\nUser question:\n{query}"
        )
    else:
        current_message = query
    messages.append({"role": "user", "content": current_message})
    return messages


def _completion_options(settings: Settings) -> dict[str, Any]:
    if settings.litellm_proxy_api_key:
        return {
            "model": settings.openai_chat_model,
            "api_key": settings.litellm_proxy_api_key,
            "api_base": settings.litellm_proxy_api_base,
        }
    return {
        "model": f"openai/{settings.openai_chat_model}",
        "api_key": settings.require_openai(),
    }


def answer_chat(
    session_id: str,
    previous_interaction_id: str | None,
    query: str,
) -> tuple[str, str]:
    settings = get_settings()
    langfuse_public_key, langfuse_secret_key = settings.require_langfuse()
    vector_store_id = get_session_vector_store_id(session_id)
    if vector_store_id is None:
        raise LookupError(f"Session {session_id} was not found")

    history = get_interaction_history(session_id, previous_interaction_id)
    openai_client = OpenAI(api_key=settings.require_openai())
    langfuse_client = Langfuse(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )

    try:
        retrieved_context = _extract_context(vector_store_id, query, openai_client)
        messages = _build_messages(query, history, retrieved_context)
        model_options = _completion_options(settings)
        with langfuse_client.start_as_current_observation(
            as_type="generation",
            name="chat-completion",
            input=messages,
            model=model_options["model"],
            metadata={
                "session_id": session_id,
                "previous_interaction_id": previous_interaction_id,
                "vector_store_id": vector_store_id,
            },
        ) as generation:
            response = litellm.completion(
                **model_options,
                messages=messages,
            )
            choices = getattr(response, "choices", None)
            if not choices:
                raise RuntimeError("LiteLLM returned no completion choices")
            message = getattr(choices[0], "message", None)
            answer = getattr(message, "content", None)
            interaction_id = getattr(response, "id", None)
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("LiteLLM returned an empty or unsupported response")
            if not isinstance(interaction_id, str) or not interaction_id:
                raise RuntimeError("LiteLLM response did not include an interaction ID")
            generation.update(output=answer)

        conversation_id = store_interaction(
            session_id=session_id,
            previous_interaction_id=previous_interaction_id,
            interaction_id=interaction_id,
            user_message=query,
            system_message=answer,
        )
        return conversation_id, answer
    finally:
        langfuse_client.flush()
        openai_client.close()
