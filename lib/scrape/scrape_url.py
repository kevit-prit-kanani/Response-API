import logging
import re
from collections import deque
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests
from openai import OpenAI
from zenrows import ZenRowsClient

from lib.config.settings import get_settings
from lib.db.ingect_to_vector_store import (
    SessionMapping,
    get_or_create_session,
    get_vector_store_source_hashes,
    ingest_markdown,
    source_url_hash,
)

logger = logging.getLogger(__name__)

MAX_CRAWL_DEPTH = 2
_MARKDOWN_LINK = re.compile(r"""!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))""")


@dataclass(frozen=True)
class ScrapeSummary:
    session_id: str
    vector_store_id: str
    pages_scraped: int
    pages_uploaded: int
    pages_skipped: int


def _normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("URL must be an absolute HTTP or HTTPS URL")
    if parts.username is not None or parts.password is not None:
        raise ValueError("URLs containing credentials are not allowed")

    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc

    hostname = parts.hostname.lower()
    if ":" in hostname:
        hostname = f"[{hostname}]"
    default_port = 443 if scheme == "https" else 80
    netloc = hostname if port in {None, default_port} else f"{hostname}:{port}"
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, parts.query, ""))


def _origin(url: str) -> tuple[str, str, int]:
    parts = urlsplit(url)
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return parts.scheme, parts.hostname or "", port


def _markdown_links(markdown: str, page_url: str) -> list[str]:
    links: list[str] = []
    for match in _MARKDOWN_LINK.finditer(markdown):
        href = match.group(1) or match.group(2)
        if href:
            try:
                links.append(_normalize_url(urljoin(page_url, href)))
            except ValueError:
                continue
    return links


def _fetch_markdown(client: ZenRowsClient, url: str) -> str:
    response = client.fetch(
        url,
        params={"response_type": "markdown"},
        timeout=90,
    )
    response.raise_for_status()
    return response.text


def scrape_url(url: str, session_id: str | None = None) -> ScrapeSummary:
    start_url = _normalize_url(url)
    settings = get_settings()
    openai_api_key = settings.require_openai()
    zenrows_api_key = settings.require_zenrows()
    zenrows_client = ZenRowsClient(zenrows_api_key, retries=2)
    openai_client = OpenAI(api_key=openai_api_key)

    try:
        session: SessionMapping = get_or_create_session(session_id, openai_client)
        existing_hashes = get_vector_store_source_hashes(
            session.vector_store_id,
            openai_client,
        )
        visited: set[str] = set()
        pending = deque([(start_url, 0)])
        pages_scraped = 0
        pages_uploaded = 0
        pages_skipped = 0
        start_origin = _origin(start_url)

        while pending:
            page_url, depth = pending.popleft()
            if page_url in visited:
                continue
            visited.add(page_url)

            page_hash = source_url_hash(page_url)
            try:
                markdown = _fetch_markdown(zenrows_client, page_url)
            except requests.RequestException:
                logger.exception("ZenRows failed for source hash %s", page_hash)
                raise
            pages_scraped += 1

            if markdown.strip():
                if ingest_markdown(
                    session.vector_store_id,
                    page_url,
                    markdown,
                    openai_client,
                    existing_hashes,
                ):
                    pages_uploaded += 1
                else:
                    pages_skipped += 1
            else:
                pages_skipped += 1
                logger.warning(
                    "ZenRows returned empty Markdown for source hash %s",
                    page_hash,
                )

            if depth < MAX_CRAWL_DEPTH:
                for linked_url in _markdown_links(markdown, page_url):
                    if _origin(linked_url) == start_origin and linked_url not in visited:
                        pending.append((linked_url, depth + 1))

        logger.info(
            "Completed crawl for session %s: scraped=%d uploaded=%d skipped=%d",
            session.session_id,
            pages_scraped,
            pages_uploaded,
            pages_skipped,
        )
        return ScrapeSummary(
            session_id=session.session_id,
            vector_store_id=session.vector_store_id,
            pages_scraped=pages_scraped,
            pages_uploaded=pages_uploaded,
            pages_skipped=pages_skipped,
        )
    finally:
        zenrows_client.close()
        openai_client.close()