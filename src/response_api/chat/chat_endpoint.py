import logging

from fastapi import APIRouter, HTTPException
from litellm.exceptions import APIError
from openai import OpenAIError
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from lib.chat.chat_service import answer_chat
from lib.config.settings import ConfigurationError
from lib.db.chat_repository import ConversationConflictError

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    previous_interaction_id: str | None = Field(default=None, min_length=1)
    query: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    previous_interaction_id: str
    output: str


@chat_router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        conversation_id, output = answer_chat(
            session_id=request.session_id,
            previous_interaction_id=request.previous_interaction_id,
            query=request.query,
        )
        return ChatResponse(
            session_id=request.session_id,
            previous_interaction_id=conversation_id,
            output=output,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="previous_interaction_id was not found for this session",
        ) from exc
    except ConversationConflictError as exc:
        logger.warning("Concurrent or duplicate chat update rejected: %s", exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ConfigurationError as exc:
        logger.error("Required chat configuration is missing: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PyMongoError as exc:
        logger.exception("MongoDB chat storage failed")
        raise HTTPException(
            status_code=503, detail="Chat storage is unavailable"
        ) from exc
    except (OpenAIError, APIError) as exc:
        logger.exception("Chat provider request failed")
        raise HTTPException(
            status_code=502,
            detail="The chat provider request failed.",
        ) from exc
    except RuntimeError as exc:
        logger.exception("Chat request failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
