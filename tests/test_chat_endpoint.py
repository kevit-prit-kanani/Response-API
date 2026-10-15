import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import main
from lib.db.chat_repository import ConversationConflictError


class ChatEndpointTests(unittest.TestCase):
    def test_chat_endpoint_returns_conversation_id_as_previous_interaction_id(self) -> None:
        with patch(
            "response_api.chat.chat_endpoint.answer_chat",
            return_value=("conversation-id", "The answer"),
        ) as answer:
            response = TestClient(main).post(
                "/chat",
                json={
                    "session_id": "session-1",
                    "previous_interaction_id": None,
                    "query": "What is on this website?",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "session_id": "session-1",
                "previous_interaction_id": "conversation-id",
                "output": "The answer",
            },
        )
        answer.assert_called_once_with(
            session_id="session-1",
            previous_interaction_id=None,
            query="What is on this website?",
        )

    def test_unknown_previous_interaction_returns_404(self) -> None:
        with patch(
            "response_api.chat.chat_endpoint.answer_chat",
            side_effect=KeyError("unknown"),
        ):
            response = TestClient(main).post(
                "/chat",
                json={
                    "session_id": "session-1",
                    "previous_interaction_id": "unknown",
                    "query": "Continue",
                },
            )

        self.assertEqual(response.status_code, 404)

    def test_stale_concurrent_interaction_returns_conflict(self) -> None:
        with patch(
            "response_api.chat.chat_endpoint.answer_chat",
            side_effect=ConversationConflictError("Conversation advanced"),
        ):
            response = TestClient(main).post(
                "/chat",
                json={
                    "session_id": "session-1",
                    "previous_interaction_id": "stale-id",
                    "query": "Continue",
                },
            )

        self.assertEqual(response.status_code, 409)


if __name__ == "__main__":
    unittest.main()
