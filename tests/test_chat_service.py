import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from lib.chat.chat_service import SYSTEM_PROMPT, answer_chat
from lib.config.settings import Settings


class ChatServiceTests(unittest.TestCase):
    def _setup_clients(self) -> tuple[Mock, Mock]:
        openai_client = Mock()
        openai_client.vector_stores.search.return_value.data = [
            Mock(content=[Mock(text="Relevant website text")])
        ]
        langfuse_client = Mock()
        observation = Mock()
        observation_context = MagicMock()
        observation_context.__enter__.return_value = observation
        langfuse_client.start_as_current_observation.return_value = observation_context
        return openai_client, langfuse_client

    def test_continuation_uses_stored_response_id_without_loading_history(self) -> None:
        settings = Settings(
            openai_api_key="test-openai-key",
            openai_chat_model="gpt-test",
            langfuse_public_key="test-public",
            langfuse_secret_key="test-secret",
        )
        openai_client, langfuse_client = self._setup_clients()
        response = SimpleNamespace(
            id="resp-provider-id",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[SimpleNamespace(type="output_text", text="The answer")],
                )
            ],
        )

        with (
            patch("lib.chat.chat_service.get_settings", return_value=settings),
            patch(
                "lib.chat.chat_service.get_session_vector_store_id",
                return_value="vs-session",
            ),
            patch(
                "lib.chat.chat_service.get_latest_response_id",
                return_value="resp-previous-id",
            ) as get_latest,
            patch("lib.chat.chat_service.OpenAI", return_value=openai_client),
            patch("lib.chat.chat_service.Langfuse", return_value=langfuse_client),
            patch("lib.chat.chat_service.litellm.responses", return_value=response) as llm,
            patch(
                "lib.chat.chat_service.store_interaction",
                return_value="previous-id",
            ) as store,
        ):
            conversation_id, output = answer_chat(
                "session-1",
                "previous-id",
                "Current question",
            )

        self.assertEqual(conversation_id, "previous-id")
        self.assertEqual(output, "The answer")
        get_latest.assert_called_once_with("session-1", "previous-id")
        openai_client.vector_stores.search.assert_called_once_with(
            vector_store_id="vs-session",
            query="Current question",
            max_num_results=5,
        )
        llm.assert_called_once()
        request = llm.call_args.kwargs
        self.assertEqual(request["previous_response_id"], "resp-previous-id")
        self.assertEqual(request["input"].splitlines()[-1], "Current question")
        self.assertIn("Relevant website text", request["input"])
        self.assertEqual(request["instructions"], SYSTEM_PROMPT)
        self.assertTrue(request["store"])
        self.assertNotIn("messages", request)
        store.assert_called_once_with(
            session_id="session-1",
            previous_interaction_id="previous-id",
            interaction_id="resp-provider-id",
            user_message="Current question",
            system_message="The answer",
            expected_previous_response_id="resp-previous-id",
        )
        langfuse_client.start_as_current_observation.assert_called_once()
        observation = (
            langfuse_client.start_as_current_observation.return_value.__enter__.return_value
        )
        observation.update.assert_called_once_with(output="The answer")
        langfuse_client.flush.assert_called_once()
        openai_client.close.assert_called_once()

    def test_new_chat_starts_response_chain_without_previous_response_id(self) -> None:
        settings = Settings(
            openai_api_key="test-openai-key",
            langfuse_public_key="test-public",
            langfuse_secret_key="test-secret",
        )
        openai_client, langfuse_client = self._setup_clients()
        openai_client.vector_stores.search.return_value.data = []
        response = SimpleNamespace(id="resp-new-id", output_text="Answer")
        stable_conversation_id = "generated-conversation-id"

        with (
            patch("lib.chat.chat_service.get_settings", return_value=settings),
            patch(
                "lib.chat.chat_service.get_session_vector_store_id",
                return_value="vs-session",
            ),
            patch("lib.chat.chat_service.OpenAI", return_value=openai_client),
            patch("lib.chat.chat_service.Langfuse", return_value=langfuse_client),
            patch("lib.chat.chat_service.litellm.responses", return_value=response) as llm,
            patch(
                "lib.chat.chat_service.store_interaction",
                return_value=stable_conversation_id,
            ) as store,
        ):
            result = answer_chat("session-1", None, "New chat question")

        self.assertEqual(result, (stable_conversation_id, "Answer"))
        self.assertEqual(
            llm.call_args.kwargs["input"],
            "New chat question",
        )
        self.assertNotIn("previous_response_id", llm.call_args.kwargs)
        store.assert_called_once_with(
            session_id="session-1",
            previous_interaction_id=None,
            interaction_id="resp-new-id",
            user_message="New chat question",
            system_message="Answer",
            expected_previous_response_id=None,
        )


if __name__ == "__main__":
    unittest.main()
