from fastapi import APIRouter

from lib.config.logging_config import configure_logging
from response_api.chat.chat_endpoint import chat_router
from response_api.ingesion.ingect_url import ingest_url_router

configure_logging()

app_router = APIRouter()
app_router.include_router(ingest_url_router)
app_router.include_router(chat_router)
