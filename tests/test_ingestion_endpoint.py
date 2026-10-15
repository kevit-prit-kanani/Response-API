import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import main
from lib.scrape.scrape_url import ScrapeSummary


class IngestionEndpointTests(unittest.TestCase):
    def test_scrape_endpoint_accepts_session_and_returns_mapping(self) -> None:
        response_data = ScrapeSummary(
            session_id="chat-1",
            vector_store_id="vs-1",
            pages_scraped=2,
            pages_uploaded=2,
            pages_skipped=0,
        )

        with patch(
            "response_api.ingesion.ingect_url.scrape_url",
            return_value=response_data,
        ) as scrape:
            response = TestClient(main).post(
                "/ingestion/scrape?session_id=chat-1",
                json="https://example.com/",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "session_id": "chat-1",
                "vector_store_id": "vs-1",
                "pages_scraped": 2,
                "pages_uploaded": 2,
                "pages_skipped": 0,
            },
        )
        scrape.assert_called_once_with("https://example.com/", "chat-1")


if __name__ == "__main__":
    unittest.main()
