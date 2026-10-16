import unittest
from datetime import UTC, datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import main
from lib.db.chat_repository import ConversationConflictError


class ChatEndpointTests(unittest.TestCase):
    def test_chat_ids_endpoint_returns_session_conversation_pairs(self) -> None:
        with patch(
            "response_api.chat.chat_endpoint.list_chat_ids",
            return_value=[
                {
                    "session_id": "session-1",
                    "previous_interaction_id": "conversation-1",
                },
                {
                    "session_id": "session-1",
                    "previous_interaction_id": "conversation-2",
                },
            ],
        ):
            response = TestClient(main).get("/chat/ids")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
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

    def test_get_chat_endpoint_returns_conversation_fields(self) -> None:
        created_at = datetime(2026, 10, 16, tzinfo=UTC)
        with patch(
            "response_api.chat.chat_endpoint.get_chat_conversation",
            return_value={
                "session_id": "session-1",
                "previous_interaction_id": "conversation-1",
                "interactions": [{"user": "hello", "system": "Hi"}],
                "created_at": created_at,
            },
        ) as get_conversation:
            response = TestClient(main).get(
                "/chat/conversation",
                params={
                    "session_id": "session-1",
                    "previous_interaction_id": "conversation-1",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "session_id": "session-1",
                "previous_interaction_id": "conversation-1",
                "interactions": [{"user": "hello", "system": "Hi"}],
                "created_at": "2026-10-16T00:00:00Z",
            },
        )
        get_conversation.assert_called_once_with("session-1", "conversation-1")

    def test_get_chat_endpoint_returns_not_found_for_unknown_pair(self) -> None:
        with patch(
            "response_api.chat.chat_endpoint.get_chat_conversation",
            return_value=None,
        ):
            response = TestClient(main).get(
                "/chat/conversation",
                params={
                    "session_id": "session-1",
                    "previous_interaction_id": "unknown",
                },
            )

        self.assertEqual(response.status_code, 404)

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
