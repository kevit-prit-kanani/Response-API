import unittest
from datetime import UTC, datetime
from unittest.mock import Mock, patch

from lib.db.chat_repository import (
    ConversationConflictError,
    get_chat_conversation,
    get_latest_response_id,
    list_chat_ids,
    store_interaction,
)


class ChatRepositoryTests(unittest.TestCase):
    def test_gets_latest_provider_response_id_without_loading_history(self) -> None:
        collection = Mock()
        collection.find_one.return_value = {
            "interaction_id": "resp_latest",
        }

        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            interaction_id = get_latest_response_id("session-1", "conversation-root")

        self.assertEqual(interaction_id, "resp_latest")
        collection.find_one.assert_called_once_with({
            "session_id": "session-1",
            "previous_interaction_id": "conversation-root",
        }, {"_id": 0, "interaction_id": 1})

    def test_rejects_legacy_chat_completion_id_for_responses_continuation(self) -> None:
        collection = Mock()
        collection.find_one.return_value = {"interaction_id": "chatcmpl-legacy"}

        with (
            patch(
                "lib.db.chat_repository._get_chat_collection",
                return_value=collection,
            ),
            self.assertRaisesRegex(
                ConversationConflictError,
                "Start a new conversation",
            ),
        ):
            get_latest_response_id("session-1", "conversation-root")

    def test_new_conversation_inserts_one_document_with_generated_root_id(self) -> None:
        collection = Mock()
        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            conversation_id = store_interaction(
                "session-1",
                None,
                "response-id-1",
                "question one",
                "answer one",
                expected_previous_response_id=None,
            )

        stored = collection.insert_one.call_args.args[0]
        self.assertEqual(stored["session_id"], "session-1")
        self.assertTrue(stored["previous_interaction_id"])
        self.assertNotEqual(stored["previous_interaction_id"], "response-id-1")
        self.assertEqual(conversation_id, stored["previous_interaction_id"])
        self.assertEqual(stored["interaction_id"], "response-id-1")
        self.assertEqual(
            stored["interactions"],
            [{"user": "question one", "system": "answer one"}],
        )
        collection.update_one.assert_not_called()

    def test_followup_updates_same_document_and_appends_turn(self) -> None:
        collection = Mock()
        collection.find_one.return_value = {
            "_id": "document-id",
            "session_id": "session-1",
            "previous_interaction_id": "conversation-root",
            "interaction_id": "response-id-1",
        }
        collection.update_one.return_value.matched_count = 1

        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            conversation_id = store_interaction(
                "session-1",
                "conversation-root",
                "response-id-2",
                "question two",
                "answer two",
                expected_previous_response_id="response-id-1",
            )

        collection.insert_one.assert_not_called()
        collection.update_one.assert_called_once_with(
            {
                "session_id": "session-1",
                "previous_interaction_id": "conversation-root",
                "interaction_id": "response-id-1",
            },
            {
                "$set": {
                    "interaction_id": "response-id-2",
                },
                "$push": {
                    "interactions": {
                        "user": "question two",
                        "system": "answer two",
                    }
                },
            },
        )
        self.assertEqual(conversation_id, "conversation-root")

    def test_followup_rejects_stale_interaction_id(self) -> None:
        collection = Mock()
        collection.find_one.return_value = {
            "_id": "document-id",
            "previous_interaction_id": "conversation-root",
            "interaction_id": "response-id-1",
        }
        collection.update_one.return_value.matched_count = 0

        with (
            patch(
                "lib.db.chat_repository._get_chat_collection",
                return_value=collection,
            ),
            self.assertRaises(ConversationConflictError),
        ):
            store_interaction(
                "session-1",
                "conversation-root",
                "new-response-id",
                "question",
                "answer",
                expected_previous_response_id="response-id-1",
            )

    def test_followup_returns_conversation_id_and_uses_latest_llm_id(self) -> None:
        collection = Mock()
        collection.find_one.return_value = {
            "_id": "legacy-document",
            "interaction_id": "legacy-response-id",
        }
        collection.update_one.return_value.matched_count = 1

        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            conversation_id = store_interaction(
                "session-1",
                "legacy-conversation-id",
                "new-response-id",
                "next question",
                "next answer",
                expected_previous_response_id="legacy-response-id",
            )

        self.assertEqual(conversation_id, "legacy-conversation-id")
        update = collection.update_one.call_args.args[1]
        self.assertEqual(update["$set"]["interaction_id"], "new-response-id")
        self.assertEqual(
            update["$push"]["interactions"],
            {"user": "next question", "system": "next answer"},
        )
        query = collection.update_one.call_args.args[0]
        self.assertEqual(query["session_id"], "session-1")
        self.assertEqual(query["previous_interaction_id"], "legacy-conversation-id")
        self.assertEqual(query["interaction_id"], "legacy-response-id")

    def test_lists_session_and_conversation_ids(self) -> None:
        collection = Mock()
        cursor = collection.find.return_value.sort.return_value
        cursor.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "session_id": "session-1",
                        "previous_interaction_id": "conversation-1",
                    },
                    {
                        "session_id": "session-1",
                        "previous_interaction_id": "conversation-2",
                    },
                ]
            )
        )

        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            chat_ids = list_chat_ids()

        self.assertEqual(
            chat_ids,
            [
                {
                    "session_id": "session-1",
                    "previous_interaction_id": "conversation-1",
                },
                {
                    "session_id": "session-1",
                    "previous_interaction_id": "conversation-2",
                },
            ],
        )
        collection.find.assert_called_once_with(
            {},
            {"_id": 0, "session_id": 1, "previous_interaction_id": 1},
        )

    def test_gets_conversation_by_session_and_stable_conversation_id(self) -> None:
        created_at = datetime(2026, 10, 16, tzinfo=UTC)
        collection = Mock()
        collection.find_one.return_value = {
            "session_id": "session-1",
            "previous_interaction_id": "conversation-1",
            "interactions": [{"user": "hello", "system": "Hi"}],
            "created_at": created_at,
        }

        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            conversation = get_chat_conversation("session-1", "conversation-1")

        self.assertEqual(
            conversation,
            {
                "session_id": "session-1",
                "previous_interaction_id": "conversation-1",
                "interactions": [{"user": "hello", "system": "Hi"}],
                "created_at": created_at,
            },
        )
        collection.find_one.assert_called_once_with(
            {
                "session_id": "session-1",
                "previous_interaction_id": "conversation-1",
            },
            {
                "_id": 0,
                "session_id": 1,
                "previous_interaction_id": 1,
                "interactions": 1,
                "created_at": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
