# URL ingestion API

The `POST /ingestion/scrape` endpoint accepts an absolute URL as a JSON string
and optionally a `session_id` query parameter. A new session gets a new OpenAI
vector store; subsequent requests with that session ID reuse the stored
vector-store mapping in MongoDB's `sessions` collection.

The crawler requests ZenRows Markdown and follows same-origin links up to two
link levels from the submitted URL. Each page is uploaded directly to the
session's vector store; page contents are not written to local files or
MongoDB. A SHA-256 URL fingerprint in each vector-store file's attributes
prevents a page already attached to that store from being uploaded again.
Runtime logs are written to `logs/response_api.log`.

Configure `ZENROWS_API_KEY`, `OPENAI_API_KEY`, and `MONGODB_URI` in the
environment (or `.env`). `MONGODB_DB` and `MONGODB_SESSIONS_COLLECTION` are
optional and default to `RAG_assessment` and `sessions`.

## Chat API

`POST /chat` accepts JSON with `session_id`, `previous_interaction_id` (nullable),
and `query`. The `session_id` must refer to a scraped session. Relevant content
is searched in its vector store and, when a previous interaction ID is supplied,
the endpoint loads the session-scoped conversation history. LiteLLM generates
the answer and its response ID becomes the returned `interaction_id`. Each
successful user/assistant turn is stored in the configured MongoDB `chat`
collection as an interaction object with `user` and `system` fields. Each
interaction record stores its parent ID, allowing branches and multiple chats
under one session.

Chat requests require `OPENAI_API_KEY`, `MONGODB_URI`,
`LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_SECRET_KEY`. Configure
`OPENAI_CHAT_MODEL` for the LiteLLM model. To route through a LiteLLM proxy,
also configure `LITELLM_PROXY_API_KEY` and `LITELLM_PROXY_API_BASE`.