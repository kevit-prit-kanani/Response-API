import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any
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
MAX_LINKS_PER_PAGE = 2
MAX_PAGES_PER_CRAWL = 20
_MARKDOWN_LINK = re.compile(r"""!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))""")
_SENSITIVE_PARAMETER = re.compile(
    r"""(?i)(apikey|api_key|token|secret)(\s*[:=]\s*)([^&\s,}"']+)"""
)


class ZenRowsError(RuntimeError):
    pass


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


def _same_origin_links_to_crawl(
    markdown: str,
    page_url: str,
    start_origin: tuple[str, str, int],
    known_urls: set[str],
) -> list[str]:
    selected: list[str] = []
    selected_urls: set[str] = set()
    for linked_url in _markdown_links(markdown, page_url):
        if (
            _origin(linked_url) != start_origin
            or linked_url in known_urls
            or linked_url in selected_urls
        ):
            continue
        selected.append(linked_url)
        selected_urls.add(linked_url)
        if len(selected) == MAX_LINKS_PER_PAGE:
            break
    return selected


def _fetch_markdown(client: ZenRowsClient, url: str) -> str:
    try:
        response = client.fetch(
            url,
            params={"response_type": "markdown"},
            timeout=90,
        )
    except requests.RequestException as exc:
        error_type = type(exc).__name__
        raise ZenRowsError(
            f"ZenRows request failed before receiving a response ({error_type})."
        ) from None

    if not response.ok:
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        detail: Any = payload
        if isinstance(payload, dict):
            detail = payload.get("error") or payload.get("message") or payload
        if not isinstance(detail, str):
            detail = str(detail)
        detail = _SENSITIVE_PARAMETER.sub(r"\1\2[REDACTED]", detail)
        detail = detail[:500]
        raise ZenRowsError(
            f"ZenRows returned HTTP {response.status_code}: {detail}"
        ) from None

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
        enqueued: set[str] = {start_url}
        pending = deque([(start_url, 0)])
        pages_scraped = 0
        pages_uploaded = 0
        pages_skipped = 0
        start_origin = _origin(start_url)

        while pending and pages_scraped < MAX_PAGES_PER_CRAWL:
            page_url, depth = pending.popleft()
            page_hash = source_url_hash(page_url)
            try:
                markdown = _fetch_markdown(zenrows_client, page_url)
            except ZenRowsError as exc:
                logger.error("ZenRows failed for source hash %s: %s", page_hash, exc)
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
                for linked_url in _same_origin_links_to_crawl(
                    markdown,
                    page_url,
                    start_origin,
                    enqueued,
                ):
                    enqueued.add(linked_url)
                    pending.append((linked_url, depth + 1))

        if pending:
            logger.warning(
                "Stopped crawl for session %s at page limit %d; %d pages remain queued",
                session.session_id,
                MAX_PAGES_PER_CRAWL,
                len(pending),
            )

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