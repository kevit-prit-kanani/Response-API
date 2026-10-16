from typing import Any

import litellm
from langfuse import Langfuse
from openai import OpenAI

from lib.config.settings import Settings, get_settings
from lib.db.chat_repository import (
    get_latest_response_id,
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


def _build_input(query: str, retrieved_context: str) -> str:
    if not retrieved_context:
        return query
    return f"Relevant website context:\n{retrieved_context}\n\nUser question:\n{query}"


def _responses_options(settings: Settings) -> dict[str, Any]:
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


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    excerpts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if getattr(content, "type", None) == "output_text" and isinstance(text, str):
                excerpts.append(text)
    return "\n".join(excerpts).strip()


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

    previous_response_id = (
        get_latest_response_id(session_id, previous_interaction_id)
        if previous_interaction_id is not None
        else None
    )
    openai_client = OpenAI(api_key=settings.require_openai())
    langfuse_client = Langfuse(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )

    try:
        retrieved_context = _extract_context(vector_store_id, query, openai_client)
        current_input = _build_input(query, retrieved_context)
        model_options = _responses_options(settings)
        response_options: dict[str, Any] = {
            **model_options,
            "input": current_input,
            "instructions": SYSTEM_PROMPT,
            "store": True,
        }
        if previous_response_id is not None:
            response_options["previous_response_id"] = previous_response_id

        with langfuse_client.start_as_current_observation(
            as_type="generation",
            name="chat-response",
            input=current_input,
            model=model_options["model"],
            metadata={
                "session_id": session_id,
                "previous_interaction_id": previous_interaction_id,
                "vector_store_id": vector_store_id,
            },
        ) as generation:
            response = litellm.responses(**response_options)
            answer = _extract_response_text(response)
            interaction_id = getattr(response, "id", None)
            if not answer:
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
            expected_previous_response_id=previous_response_id,
        )
        return conversation_id, answer
    finally:
        langfuse_client.flush()
        openai_client.close()
