import unittest
from unittest.mock import MagicMock, Mock, patch

from lib.config.settings import Settings
from lib.db.chat_repository import Interaction
from lib.chat.chat_service import answer_chat


class ChatServiceTests(unittest.TestCase):
    def test_continuation_returns_conversation_id_and_stores_provider_id(self) -> None:
        settings = Settings(
            openai_api_key="test-openai-key",
            openai_chat_model="gpt-test",
            langfuse_public_key="test-public",
            langfuse_secret_key="test-secret",
        )
        previous_history: list[Interaction] = [
            {"user": "Earlier question", "system": "Earlier answer"}
        ]
        openai_client = Mock()
        openai_client.vector_stores.search.return_value.data = [
            Mock(content=[Mock(text="Relevant website text")])
        ]
        langfuse_client = Mock()
        observation = Mock()
        observation_context = MagicMock()
        observation_context.__enter__.return_value = observation
        langfuse_client.start_as_current_observation.return_value = observation_context
        completion = Mock(
            id="chatcmpl-provider-id",
            choices=[Mock(message=Mock(content="The answer"))],
        )

        with (
            patch("lib.chat.chat_service.get_settings", return_value=settings),
            patch(
                "lib.chat.chat_service.get_session_vector_store_id",
                return_value="vs-session",
            ),
            patch(
                "lib.chat.chat_service.get_interaction_history",
                return_value=previous_history,
            ),
            patch("lib.chat.chat_service.OpenAI", return_value=openai_client),
            patch("lib.chat.chat_service.Langfuse", return_value=langfuse_client),
            patch("lib.chat.chat_service.litellm.completion", return_value=completion) as llm,
            patch(
                "lib.chat.chat_service.store_interaction",
                return_value="previous-id",
            ) as store,
        ):
            interaction_id, output = answer_chat(
                "session-1",
                "previous-id",
                "Current question",
            )

        self.assertEqual(interaction_id, "previous-id")
        self.assertEqual(output, "The answer")
        self.assertNotIn("temperature", llm.call_args.kwargs)
        openai_client.vector_stores.search.assert_called_once_with(
            vector_store_id="vs-session",
            query="Current question",
            max_num_results=5,
        )
        messages = llm.call_args.kwargs["messages"]
        self.assertEqual(messages[1], {"role": "user", "content": "Earlier question"})
        self.assertEqual(
            messages[2],
            {"role": "assistant", "content": "Earlier answer"},
        )
        self.assertIn("Relevant website text", messages[-1]["content"])
        self.assertIn("Current question", messages[-1]["content"])
        store.assert_called_once_with(
            session_id="session-1",
            previous_interaction_id="previous-id",
            interaction_id="chatcmpl-provider-id",
            user_message="Current question",
            system_message="The answer",
        )
        observation.update.assert_called_once_with(output="The answer")
        langfuse_client.flush.assert_called_once()
        openai_client.close.assert_called_once()

    def test_new_chat_has_no_previous_interaction_context(self) -> None:
        settings = Settings(
            openai_api_key="test-openai-key",
            langfuse_public_key="test-public",
            langfuse_secret_key="test-secret",
        )
        openai_client = Mock()
        openai_client.vector_stores.search.return_value.data = []
        langfuse_client = Mock()
        observation_context = MagicMock()
        observation_context.__enter__.return_value = Mock()
        langfuse_client.start_as_current_observation.return_value = observation_context
        completion = Mock(
            id="provider-interaction-id",
            choices=[Mock(message=Mock(content="Answer"))],
        )
        stable_conversation_id = "generated-conversation-id"

        with (
            patch("lib.chat.chat_service.get_settings", return_value=settings),
            patch(
                "lib.chat.chat_service.get_session_vector_store_id",
                return_value="vs-session",
            ),
            patch(
                "lib.chat.chat_service.get_interaction_history",
                return_value=[],
            ) as load_history,
            patch("lib.chat.chat_service.OpenAI", return_value=openai_client),
            patch("lib.chat.chat_service.Langfuse", return_value=langfuse_client),
            patch("lib.chat.chat_service.litellm.completion", return_value=completion) as llm,
            patch(
                "lib.chat.chat_service.store_interaction",
                return_value=stable_conversation_id,
            ) as store,
        ):
            result = answer_chat("session-1", None, "New chat question")

        store.assert_called_once_with(
            session_id="session-1",
            previous_interaction_id=None,
            interaction_id="provider-interaction-id",
            user_message="New chat question",
            system_message="Answer",
        )
        self.assertEqual(result, (stable_conversation_id, "Answer"))
        load_history.assert_called_once_with("session-1", None)
        self.assertEqual(
            llm.call_args.kwargs["messages"][-1],
            {"role": "user", "content": "New chat question"},
        )


if __name__ == "__main__":
    unittest.main()
