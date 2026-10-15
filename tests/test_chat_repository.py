import unittest
from unittest.mock import Mock, patch

from lib.db.chat_repository import (
    ConversationConflictError,
    get_interaction_history,
    store_interaction,
)


class ChatRepositoryTests(unittest.TestCase):
    def test_loads_ordered_turns_from_current_conversation_document(self) -> None:
        collection = Mock()
        collection.find_one.return_value = {
            "session_id": "session-1",
            "interaction_id": "interaction-2",
            "previous_interaction_id": "conversation-root",
            "interactions": [
                {"user": "First question", "system": "First answer"},
                {"user": "Second question", "system": "Second answer"},
            ],
        }

        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            history = get_interaction_history("session-1", "conversation-root")

        self.assertEqual(
            history,
            [
                {"user": "First question", "system": "First answer"},
                {"user": "Second question", "system": "Second answer"},
            ],
        )
        collection.find_one.assert_called_once_with({
            "session_id": "session-1",
            "previous_interaction_id": "conversation-root",
        })

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
            )

        collection.insert_one.assert_not_called()
        collection.update_one.assert_called_once_with(
            {
                "_id": "document-id",
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

    def test_new_chat_starts_with_empty_history(self) -> None:
        self.assertEqual(get_interaction_history("session-1", None), [])


if __name__ == "__main__":
    unittest.main()
