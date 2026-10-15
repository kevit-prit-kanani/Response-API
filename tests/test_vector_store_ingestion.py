import unittest
from unittest.mock import Mock, patch

from lib.db.ingect_to_vector_store import (
    SessionMapping,
    get_or_create_session,
    get_vector_store_source_hashes,
    ingest_markdown,
    source_url_hash,
)


class VectorStoreIngestionTests(unittest.TestCase):
    def test_existing_session_reuses_vector_store(self) -> None:
        collection = Mock()
        collection.find_one.return_value = {
            "session_id": "chat-1",
            "vector_store_id": "vs-existing",
        }
        openai_client = Mock()

        with patch(
            "lib.db.ingect_to_vector_store._get_sessions_collection",
            return_value=collection,
        ):
            session = get_or_create_session("chat-1", openai_client)

        self.assertEqual(session, SessionMapping("chat-1", "vs-existing"))
        openai_client.vector_stores.create.assert_not_called()

    def test_new_session_persists_its_vector_store_mapping(self) -> None:
        collection = Mock()
        collection.find_one.return_value = None
        openai_client = Mock()
        openai_client.vector_stores.create.return_value.id = "vs-new"

        with patch(
            "lib.db.ingect_to_vector_store._get_sessions_collection",
            return_value=collection,
        ):
            session = get_or_create_session("chat-new", openai_client)

        self.assertEqual(session, SessionMapping("chat-new", "vs-new"))
        document = collection.insert_one.call_args.args[0]
        self.assertEqual(document["session_id"], "chat-new")
        self.assertEqual(document["vector_store_id"], "vs-new")

    def test_existing_source_hash_is_not_uploaded_again(self) -> None:
        page_url = "https://example.com/page"
        openai_client = Mock()

        uploaded = ingest_markdown(
            "vs-1",
            page_url,
            "Page Markdown",
            openai_client,
            {source_url_hash(page_url)},
        )

        self.assertFalse(uploaded)
        openai_client.vector_stores.files.upload_and_poll.assert_not_called()

    def test_successful_upload_records_hash_after_processing(self) -> None:
        page_url = "https://example.com/page"
        openai_client = Mock()
        openai_client.vector_stores.files.upload_and_poll.return_value.status = (
            "completed"
        )
        known_hashes: set[str] = set()

        uploaded = ingest_markdown(
            "vs-1",
            page_url,
            "# Page",
            openai_client,
            known_hashes,
        )

        self.assertTrue(uploaded)
        self.assertIn(source_url_hash(page_url), known_hashes)
        kwargs = openai_client.vector_stores.files.upload_and_poll.call_args.kwargs
        self.assertEqual(kwargs["vector_store_id"], "vs-1")
        self.assertEqual(kwargs["attributes"]["source_url_hash"], source_url_hash(page_url))
        self.assertEqual(kwargs["file"][0], f"page-{source_url_hash(page_url)}.md")

    def test_failed_upload_does_not_mark_url_as_ingested(self) -> None:
        page_url = "https://example.com/page"
        openai_client = Mock()
        openai_client.vector_stores.files.upload_and_poll.return_value.status = "failed"
        known_hashes: set[str] = set()

        with self.assertRaises(RuntimeError):
            ingest_markdown(
                "vs-1",
                page_url,
                "# Page",
                openai_client,
                known_hashes,
            )

        self.assertNotIn(source_url_hash(page_url), known_hashes)

    def test_reads_source_hashes_from_vector_store_file_attributes(self) -> None:
        expected_hash = "page-hash"
        openai_client = Mock()
        openai_client.vector_stores.files.list.return_value = [
            Mock(status="completed", attributes={"source_url_hash": expected_hash}),
            Mock(status="completed", attributes=None),
            Mock(status="failed", attributes={"source_url_hash": "failed-hash"}),
        ]

        hashes = get_vector_store_source_hashes("vs-1", openai_client)

        self.assertEqual(hashes, {expected_hash})


if __name__ == "__main__":
    unittest.main()
