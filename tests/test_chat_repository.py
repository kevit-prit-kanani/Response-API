import unittest
from unittest.mock import Mock, patch

from lib.db.chat_repository import get_interaction_history, store_interaction


class ChatRepositoryTests(unittest.TestCase):
    def test_loads_history_from_previous_interaction_chain(self) -> None:
        collection = Mock()
        collection.find_one.side_effect = [
            {
                "session_id": "session-1",
                "interaction_id": "interaction-2",
                "previous_interaction_id": "interaction-1",
                "interactions": [{"user": "Second question", "system": "Second answer"}],
            },
            {
                "session_id": "session-1",
                "interaction_id": "interaction-1",
                "previous_interaction_id": None,
                "interactions": [{"user": "First question", "system": "First answer"}],
            },
        ]

        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            history = get_interaction_history("session-1", "interaction-2")

        self.assertEqual(
            history,
            [
                {"user": "First question", "system": "First answer"},
                {"user": "Second question", "system": "Second answer"},
            ],
        )
        self.assertEqual(collection.find_one.call_args_list[0].args[0]["session_id"], "session-1")

    def test_stores_new_llm_response_id_and_pair(self) -> None:
        collection = Mock()
        with patch(
            "lib.db.chat_repository._get_chat_collection",
            return_value=collection,
        ):
            store_interaction(
                "session-1",
                "parent-id",
                "response-id",
                "question",
                "answer",
            )

        stored = collection.insert_one.call_args.args[0]
        self.assertEqual(stored["session_id"], "session-1")
        self.assertEqual(stored["previous_interaction_id"], "parent-id")
        self.assertEqual(stored["interaction_id"], "response-id")
        self.assertEqual(
            stored["interactions"],
            [{"user": "question", "system": "answer"}],
        )

    def test_new_chat_starts_with_empty_history(self) -> None:
        self.assertEqual(get_interaction_history("session-1", None), [])


if __name__ == "__main__":
    unittest.main()
