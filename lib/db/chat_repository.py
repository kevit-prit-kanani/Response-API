import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import TypedDict
from uuid import uuid4

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from lib.config.settings import get_settings

logger = logging.getLogger(__name__)


class Interaction(TypedDict):
    user: str
    system: str


class ChatId(TypedDict):
    session_id: str
    previous_interaction_id: str | None


class ChatConversation(TypedDict):
    session_id: str
    previous_interaction_id: str | None
    interactions: list[Interaction]
    created_at: datetime


class ConversationConflictError(RuntimeError):
    pass


@lru_cache
def _get_chat_collection() -> Collection:
    settings = get_settings()
    client = MongoClient(settings.require_mongodb(), serverSelectionTimeoutMS=5000)
    collection = client[settings.mongodb_db][settings.mongodb_chat_collection]
    collection.create_index(
        [("session_id", ASCENDING), ("interaction_id", ASCENDING)],
        unique=True,
    )
    collection.create_index(
        [("session_id", ASCENDING), ("previous_interaction_id", ASCENDING)]
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


def list_chat_ids() -> list[ChatId]:
    documents = _get_chat_collection().find(
        {},
        {"_id": 0, "session_id": 1, "previous_interaction_id": 1},
    ).sort("created_at", ASCENDING)
    return [
        {
            "session_id": document["session_id"],
            "previous_interaction_id": document.get("previous_interaction_id"),
        }
        for document in documents
    ]


def get_chat_conversation(
    session_id: str,
    previous_interaction_id: str,
) -> ChatConversation | None:
    document = _get_chat_collection().find_one(
        {
            "session_id": session_id,
            "previous_interaction_id": previous_interaction_id,
        },
        {
            "_id": 0,
            "session_id": 1,
            "previous_interaction_id": 1,
            "interactions": 1,
            "created_at": 1,
        },
    )
    if document is None:
        return None
    return {
        "session_id": document["session_id"],
        "previous_interaction_id": document.get("previous_interaction_id"),
        "interactions": document["interactions"],
        "created_at": document["created_at"],
    }


def get_interaction_history(
    session_id: str,
    previous_interaction_id: str | None,
) -> list[Interaction]:
    if previous_interaction_id is None:
        return []

    collection = _get_chat_collection()
    document = collection.find_one(
        {
            "session_id": session_id,
            "previous_interaction_id": previous_interaction_id,
        }
    )
    if document is None:
        raise KeyError(previous_interaction_id)
    interactions = document.get("interactions")
    if not isinstance(interactions, list):
        raise RuntimeError(
            f"Chat interaction {previous_interaction_id} has invalid history"
        )

    history: list[Interaction] = []
    for interaction in interactions:
        if (
            not isinstance(interaction, dict)
            or not isinstance(interaction.get("user"), str)
            or not isinstance(interaction.get("system"), str)
        ):
            raise RuntimeError(
                f"Chat interaction {previous_interaction_id} has invalid history"
            )
        history.append(
            {
                "user": interaction["user"],
                "system": interaction["system"],
            }
        )
    return history


def store_interaction(
    session_id: str,
    previous_interaction_id: str | None,
    interaction_id: str,
    user_message: str,
    system_message: str,
) -> str:
    collection = _get_chat_collection()
    interaction = {"user": user_message, "system": system_message}

    if previous_interaction_id is None:
        conversation_id = str(uuid4())
        document = {
            "session_id": session_id,
            "previous_interaction_id": conversation_id,
            "interaction_id": interaction_id,
            "interactions": [interaction],
            "created_at": datetime.now(UTC),
        }
        try:
            collection.insert_one(document)
        except DuplicateKeyError as exc:
            raise ConversationConflictError(
                "The LLM returned an interaction ID already used in this session"
            ) from exc
    else:
        current = collection.find_one(
            {
                "session_id": session_id,
                "previous_interaction_id": previous_interaction_id,
            },
            {"_id": 1, "interaction_id": 1},
        )
        if current is None:
            raise KeyError(previous_interaction_id)
        conversation_id = previous_interaction_id

        latest_interaction_id = current.get("interaction_id")
        if not isinstance(latest_interaction_id, str) or not latest_interaction_id:
            raise RuntimeError(
                f"Conversation {previous_interaction_id} has no latest interaction ID"
            )

        try:
            result = collection.update_one(
                {
                    "_id": current["_id"],
                    "session_id": session_id,
                    "previous_interaction_id": previous_interaction_id,
                    "interaction_id": latest_interaction_id,
                },
                {
                    "$set": {
                        "interaction_id": interaction_id,
                    },
                    "$push": {"interactions": interaction},
                },
            )
        except DuplicateKeyError as exc:
            raise ConversationConflictError(
                "The LLM returned an interaction ID already used in this session"
            ) from exc
        if result.matched_count != 1:
            raise ConversationConflictError(
                "This conversation has already advanced; retry with its latest interaction ID"
            )

    logger.info(
        "Stored chat turn %s in conversation %s for session %s",
        interaction_id,
        conversation_id,
        session_id,
    )
    return conversation_id
