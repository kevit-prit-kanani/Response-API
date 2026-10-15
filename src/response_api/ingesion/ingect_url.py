import logging
from typing import Annotated

import requests
from fastapi import APIRouter, Body, HTTPException, Query
from openai import OpenAIError
from pydantic import BaseModel
from pymongo.errors import PyMongoError

from lib.config.settings import ConfigurationError
from lib.scrape.scrape_url import ScrapeSummary, scrape_url

logger = logging.getLogger(__name__)

ingest_url_router = APIRouter(prefix="/ingestion", tags=["ingestion"])


class ScrapeResponse(BaseModel):
    session_id: str
    vector_store_id: str
    pages_scraped: int
    pages_uploaded: int
    pages_skipped: int


@ingest_url_router.post("/scrape", response_model=ScrapeResponse)
def ingest_url(
    url: Annotated[str, Body(description="Absolute HTTP or HTTPS URL to crawl")],
    session_id: Annotated[
        str | None,
        Query(min_length=1, max_length=128, description="Existing chat session ID"),
    ] = None,
) -> ScrapeSummary:
    try:
        return scrape_url(url, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except PyMongoError as exc:
        logger.exception("MongoDB session persistence failed")
        raise HTTPException(
            status_code=503,
            detail="MongoDB is unavailable; the scrape could not be associated with a session.",
        ) from exc
    except ConfigurationError as exc:
        logger.error("Required service configuration is missing: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (requests.RequestException, OpenAIError, RuntimeError) as exc:
        logger.exception("URL crawl or vector-store ingestion failed")
        raise HTTPException(
            status_code=502,
            detail="The crawl or vector-store ingestion failed.",
        ) from exc
