import unittest
from unittest.mock import Mock, patch

from lib.config.settings import Settings
from lib.db.ingect_to_vector_store import SessionMapping, source_url_hash
from lib.scrape.scrape_url import (
    MAX_LINKS_PER_PAGE,
    MAX_PAGES_PER_CRAWL,
    ZenRowsError,
    _fetch_markdown,
    _markdown_links,
    _normalize_url,
    _same_origin_links_to_crawl,
    scrape_url,
)


class ScrapeUrlTests(unittest.TestCase):
    def test_normalize_url_removes_fragment_and_default_port(self) -> None:
        self.assertEqual(
            _normalize_url("HTTPS://Example.com:443/path#section"),
            "https://example.com/path",
        )

    def test_url_remains_unescaped_for_zenrows_client(self) -> None:
        client = Mock()
        response = Mock(ok=True, text="# Books")
        client.fetch.return_value = response
        url = "https://books.toscrape.com/"

        markdown = _fetch_markdown(client, url)

        self.assertEqual(markdown, "# Books")
        client.fetch.assert_called_once_with(
            url,
            params={"response_type": "markdown"},
            timeout=90,
        )

    def test_zenrows_error_includes_sanitized_detail_without_request_url(self) -> None:
        client = Mock()
        response = Mock(
            ok=False,
            status_code=400,
            text="",
        )
        response.json.return_value = {
            "error": "Invalid request for apikey=secret-value"
        }
        client.fetch.return_value = response

        with self.assertRaisesRegex(
            ZenRowsError,
            r"ZenRows returned HTTP 400: Invalid request for apikey=\[REDACTED\]",
        ) as raised:
            _fetch_markdown(client, "https://example.com/")

        self.assertNotIn("secret-value", str(raised.exception))
        self.assertNotIn("api.zenrows.com", str(raised.exception))

    def test_markdown_links_resolve_relative_targets(self) -> None:
        links = _markdown_links(
            "[relative](../child) [absolute](https://example.com/help)",
            "https://example.com/docs/index",
        )
        self.assertEqual(
            links,
            [
                "https://example.com/child",
                "https://example.com/help",
            ],
        )

    def test_selects_at_most_two_unique_same_origin_links_per_page(self) -> None:
        links = " ".join(
            f"[page {index}](https://example.com/page-{index})"
            for index in range(5)
        )
        markdown = f"[external](https://other.example/page) {links}"

        selected = _same_origin_links_to_crawl(
            markdown,
            "https://example.com/",
            ("https", "example.com", 443),
            {"https://example.com/"},
        )

        self.assertEqual(
            selected,
            [
                "https://example.com/page-0",
                "https://example.com/page-1",
            ],
        )
        self.assertEqual(len(selected), MAX_LINKS_PER_PAGE)

    def test_crawl_stops_at_twenty_pages(self) -> None:
        zenrows_client = Mock()

        def fetch(url: str, **_: object) -> Mock:
            page_number = int(url.split("p=")[-1]) if "p=" in url else 0
            children = (
                f"[left](https://example.com/?p={page_number * 2 + 1}) "
                f"[right](https://example.com/?p={page_number * 2 + 2})"
            )
            response = Mock(ok=True, text=children)
            return response

        zenrows_client.fetch.side_effect = fetch
        openai_client = Mock()
        session = SessionMapping("chat-1", "vs-1")

        with (
            patch(
                "lib.scrape.scrape_url.get_settings",
                return_value=Settings(
                    openai_api_key="test-openai-key",
                    zenrows_api_key="test-zenrows-key",
                ),
            ),
            patch("lib.scrape.scrape_url.OpenAI", return_value=openai_client),
            patch("lib.scrape.scrape_url.ZenRowsClient", return_value=zenrows_client),
            patch(
                "lib.scrape.scrape_url.get_or_create_session",
                return_value=session,
            ),
            patch(
                "lib.scrape.scrape_url.get_vector_store_source_hashes",
                return_value=set(),
            ),
            patch(
                "lib.scrape.scrape_url.ingest_markdown",
                return_value=True,
            ),
            patch("lib.scrape.scrape_url.MAX_CRAWL_DEPTH", 10),
        ):
            result = scrape_url("https://example.com/")

        self.assertEqual(MAX_PAGES_PER_CRAWL, 20)
        self.assertEqual(result.pages_scraped, MAX_PAGES_PER_CRAWL)
        self.assertEqual(zenrows_client.fetch.call_count, MAX_PAGES_PER_CRAWL)

    def test_crawls_same_origin_links_to_two_levels_and_uploads_once(self) -> None:
        pages = {
            "https://example.com/": (
                "[child](/child) [external](https://other.example/page)"
            ),
            "https://example.com/child": "[grandchild](/child/grandchild)",
            "https://example.com/child/grandchild": "Grandchild content",
        }
        zenrows_client = Mock()

        def fetch(url: str, **_: object) -> Mock:
            response = Mock()
            response.text = pages[url]
            return response

        zenrows_client.fetch.side_effect = fetch
        openai_client = Mock()
        session = SessionMapping("chat-1", "vs-1")
        uploaded_urls: list[str] = []

        def ingest(
            vector_store_id: str,
            page_url: str,
            markdown: str,
            client: Mock,
            hashes: set[str],
        ) -> bool:
            self.assertEqual(vector_store_id, "vs-1")
            self.assertIs(client, openai_client)
            self.assertTrue(markdown)
            uploaded_urls.append(page_url)
            hashes.add(source_url_hash(page_url))
            return True

        with (
            patch(
                "lib.scrape.scrape_url.get_settings",
                return_value=Settings(
                    openai_api_key="test-openai-key",
                    zenrows_api_key="test-zenrows-key",
                ),
            ),
            patch("lib.scrape.scrape_url.OpenAI", return_value=openai_client),
            patch("lib.scrape.scrape_url.ZenRowsClient", return_value=zenrows_client),
            patch(
                "lib.scrape.scrape_url.get_or_create_session",
                return_value=session,
            ),
            patch(
                "lib.scrape.scrape_url.get_vector_store_source_hashes",
                return_value=set(),
            ),
            patch("lib.scrape.scrape_url.ingest_markdown", side_effect=ingest),
        ):
            result = scrape_url("https://example.com/")

        self.assertEqual(result.session_id, "chat-1")
        self.assertEqual(result.vector_store_id, "vs-1")
        self.assertEqual(result.pages_scraped, 3)
        self.assertEqual(result.pages_uploaded, 3)
        self.assertEqual(result.pages_skipped, 0)
        self.assertEqual(len(uploaded_urls), 3)
        self.assertNotIn("https://other.example/page", uploaded_urls)
        self.assertEqual(zenrows_client.fetch.call_count, 3)
        zenrows_client.close.assert_called_once()
        openai_client.close.assert_called_once()

    def test_does_not_upload_url_already_attached_to_session_store(self) -> None:
        page_url = "https://example.com/"
        zenrows_client = Mock()
        response = Mock()
        response.text = "Previously ingested content"
        zenrows_client.fetch.return_value = response
        openai_client = Mock()

        with (
            patch(
                "lib.scrape.scrape_url.get_settings",
                return_value=Settings(
                    openai_api_key="test-openai-key",
                    zenrows_api_key="test-zenrows-key",
                ),
            ),
            patch("lib.scrape.scrape_url.OpenAI", return_value=openai_client),
            patch("lib.scrape.scrape_url.ZenRowsClient", return_value=zenrows_client),
            patch(
                "lib.scrape.scrape_url.get_or_create_session",
                return_value=SessionMapping("chat-1", "vs-1"),
            ),
            patch(
                "lib.scrape.scrape_url.get_vector_store_source_hashes",
                return_value={source_url_hash(page_url)},
            ),
            patch(
                "lib.scrape.scrape_url.ingest_markdown",
                return_value=False,
            ) as upload,
        ):
            result = scrape_url(page_url, session_id="chat-1")

        self.assertEqual(result.pages_scraped, 1)
        self.assertEqual(result.pages_uploaded, 0)
        self.assertEqual(result.pages_skipped, 1)
        upload.assert_called_once()
        self.assertIn(source_url_hash(page_url), upload.call_args.args[4])
        zenrows_client.close.assert_called_once()
        openai_client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
