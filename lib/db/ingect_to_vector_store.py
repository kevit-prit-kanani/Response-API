import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from uuid import uuid4

from openai import OpenAI, OpenAIError
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError

from lib.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionMapping:
    session_id: str
    vector_store_id: str


@lru_cache
def _get_sessions_collection() -> Collection:
    settings = get_settings()
    client = MongoClient(settings.require_mongodb(), serverSelectionTimeoutMS=5000)
    collection = client[settings.mongodb_db][settings.mongodb_sessions_collection]
    collection.create_index([("session_id", ASCENDING)], unique=True)
    return collection


def source_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def get_or_create_session(
    requested_session_id: str | None,
    openai_client: OpenAI,
) -> SessionMapping:
    session_id = requested_session_id or str(uuid4())
    sessions = _get_sessions_collection()
    existing = sessions.find_one({"session_id": session_id})
    if existing is not None:
        vector_store_id = existing.get("vector_store_id")
        if not isinstance(vector_store_id, str) or not vector_store_id:
            raise RuntimeError(f"Session {session_id} has no vector store mapping")
        return SessionMapping(session_id, vector_store_id)

    session_name = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    vector_store = openai_client.vector_stores.create(
        name=f"chat-session-{session_name}",
        metadata={"session_id_hash": session_name},
    )
    document = {
        "session_id": session_id,
        "vector_store_id": vector_store.id,
        "created_at": datetime.now(UTC),
    }
    try:
        sessions.insert_one(document)
    except DuplicateKeyError:
        _delete_orphan_vector_store(openai_client, vector_store.id)
        existing = sessions.find_one({"session_id": session_id})
        if existing is None or not isinstance(existing.get("vector_store_id"), str):
            raise RuntimeError(
                f"Session {session_id} was concurrently created without a vector store"
            )
        return SessionMapping(session_id, existing["vector_store_id"])
    except PyMongoError:
        _delete_orphan_vector_store(openai_client, vector_store.id)
        raise

    logger.info("Created vector store for chat session %s", session_id)
    return SessionMapping(session_id, vector_store.id)


def _delete_orphan_vector_store(openai_client: OpenAI, vector_store_id: str) -> None:
    try:
        openai_client.vector_stores.delete(vector_store_id)
    except OpenAIError:
        logger.exception("Could not remove unpaired vector store %s", vector_store_id)


def get_vector_store_source_hashes(
    vector_store_id: str,
    openai_client: OpenAI,
) -> set[str]:
    source_hashes: set[str] = set()
    files = openai_client.vector_stores.files.list(
        vector_store_id=vector_store_id,
        limit=100,
    )
    for vector_file in files:
        if vector_file.status not in {"completed", "in_progress"}:
            continue
        attributes = vector_file.attributes or {}
        source_hash = attributes.get("source_url_hash")
        if isinstance(source_hash, str):
            source_hashes.add(source_hash)
    return source_hashes


def ingest_markdown(
    vector_store_id: str,
    url: str,
    markdown: str,
    openai_client: OpenAI,
    existing_source_hashes: set[str],
) -> bool:
    source_hash = source_url_hash(url)
    if source_hash in existing_source_hashes:
        return False

    filename = f"page-{source_hash}.md"
    content = f"Source URL: {url}\n\n{markdown}".encode()
    vector_file = openai_client.vector_stores.files.upload_and_poll(
        vector_store_id=vector_store_id,
        file=(filename, content),
        attributes={"source_url_hash": source_hash},
    )
    if vector_file.status != "completed":
        raise RuntimeError(
            f"OpenAI did not complete ingestion for source hash {source_hash}"
        )

    existing_source_hashes.add(source_hash)
    logger.info(
        "Uploaded source hash %s to vector store %s",
        source_hash,
        vector_store_id,
    )
    return True
