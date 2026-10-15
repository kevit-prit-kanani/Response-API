import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import main


class ChatEndpointTests(unittest.TestCase):
    def test_chat_endpoint_returns_generated_interaction_id(self) -> None:
        with patch(
            "response_api.chat.chat_endpoint.answer_chat",
            return_value=("provider-response-id", "The answer"),
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
                "interaction_id": "provider-response-id",
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


if __name__ == "__main__":
    unittest.main()
