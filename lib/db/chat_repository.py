import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import TypedDict

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from lib.config.settings import get_settings

logger = logging.getLogger(__name__)


class Interaction(TypedDict):
    user: str
    system: str


@lru_cache
def _get_chat_collection() -> Collection:
    settings = get_settings()
    client = MongoClient(settings.require_mongodb(), serverSelectionTimeoutMS=5000)
    collection = client[settings.mongodb_db][settings.mongodb_chat_collection]
    collection.create_index(
        [("session_id", ASCENDING), ("interaction_id", ASCENDING)],
        unique=True,
    )
    return collection


@lru_cache
def _get_sessions_collection() -> Collection:
    settings = get_settings()
    client = MongoClient(settings.require_mongodb(), serverSelectionTimeoutMS=5000)
    return client[settings.mongodb_db][settings.mongodb_sessions_collection]


def get_session_vector_store_id(session_id: str) -> str | None:
    session = _get_sessions_collection().find_one(
        {"session_id": session_id}, {"vector_store_id": 1}
    )
    if session is None:
        return None
    vector_store_id = session.get("vector_store_id")
    if not isinstance(vector_store_id, str) or not vector_store_id:
        raise RuntimeError(f"Session {session_id} has no vector store mapping")
    return vector_store_id


def get_interaction_history(
    session_id: str,
    previous_interaction_id: str | None,
) -> list[Interaction]:
    if previous_interaction_id is None:
        return []

    collection = _get_chat_collection()
    interaction_id: str | None = previous_interaction_id
    visited: set[str] = set()
    history: list[Interaction] = []

    while interaction_id is not None:
        if interaction_id in visited:
            raise RuntimeError("Chat interaction history contains a cycle")
        visited.add(interaction_id)
        document = collection.find_one(
            {"session_id": session_id, "interaction_id": interaction_id}
        )
        if document is None:
            raise KeyError(previous_interaction_id)
        interactions = document.get("interactions")
        if not isinstance(interactions, list):
            raise RuntimeError(f"Chat interaction {interaction_id} has invalid history")
        for interaction in interactions:
            if (
                not isinstance(interaction, dict)
                or not isinstance(interaction.get("user"), str)
                or not isinstance(interaction.get("system"), str)
            ):
                raise RuntimeError(f"Chat interaction {interaction_id} has invalid history")
            history.append(
                {
                    "user": interaction["user"],
                    "system": interaction["system"],
                }
            )
        parent_id = document.get("previous_interaction_id")
        interaction_id = parent_id if isinstance(parent_id, str) else None

    history.reverse()
    return history


def store_interaction(
    session_id: str,
    previous_interaction_id: str | None,
    interaction_id: str,
    user_message: str,
    system_message: str,
) -> None:
    document = {
        "session_id": session_id,
        "previous_interaction_id": previous_interaction_id,
        "interaction_id": interaction_id,
        "interactions": [{"user": user_message, "system": system_message}],
        "created_at": datetime.now(UTC),
    }
    _get_chat_collection().insert_one(document)
    logger.info(
        "Stored chat interaction %s for session %s",
        interaction_id,
        session_id,
    )
