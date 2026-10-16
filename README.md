# Response API

FastAPI service for crawling websites with ZenRows, adding the resulting
Markdown pages to an OpenAI vector store, and chatting over the retrieved
website content. LiteLLM handles chat completions, Langfuse traces generations,
and MongoDB stores session/vector-store mappings and conversation histories.

## Features

- Crawl a submitted URL to depth two, following at most two same-origin links
  from each page and scraping no more than 20 pages per request.
- Reuse one OpenAI vector store for each chat session.
- Avoid adding the same page URL to a session's vector store more than once.
- Retrieve relevant vector-store content and provide it to LiteLLM as chat
  context.
- Store each conversation as a MongoDB `chat` document with an ordered
  `interactions` array.
- Trace chat generations to Langfuse.
- Allow cross-origin requests (CORS is configured to allow any origin,
  method, and header).

## Project structure

```text
Response-API/
├── lib/
│   ├── chat/
│   │   └── chat_service.py             # RAG context and LiteLLM/Langfuse
│   ├── config/
│   │   ├── db.py                       # Database configuration module
│   │   ├── logging_config.py           # Rotating logs under logs/
│   │   └── settings.py                 # Environment-backed settings
│   ├── db/
│   │   ├── chat_repository.py          # Chat/session MongoDB operations
│   │   └── ingect_to_vector_store.py   # Session/vector store and ingestion
│   └── scrape/
│       └── scrape_url.py               # ZenRows two-level crawler
├── src/
│   └── response_api/
│       ├── chat/
│       │   └── chat_endpoint.py        # Chat and conversation endpoints
│       ├── ingesion/
│       │   └── ingect_url.py           # URL ingestion endpoint
│       ├── app.py                      # API router registration
│       └── __init__.py
├── tests/                              # Unit and API tests
├── logs/                               # Runtime logs (git-ignored)
├── .env.example                        # Environment-variable template
├── main.py                             # FastAPI application entry point
├── pyproject.toml                      # Dependencies and project metadata
├── uv.lock                             # Locked dependencies
└── README.md
```

## Requirements

- Python 3.14 or newer (the project declares `requires-python = ">=3.14"`).
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python and
  dependency management.
- A ZenRows API key.
- An OpenAI API key with access to vector stores and the configured chat model.
- A MongoDB connection string.
- Langfuse public and secret keys for chat tracing.

## Download and install

Clone the GitHub repository and install the locked dependencies:

```bash
git clone https://github.com/kevit-prit-kanani/Response-API.git
cd Response-API
uv sync
```

If Python 3.14 is not installed, uv can install it:

```bash
uv python install 3.14
uv sync
```

Create a local environment file from the example:

```bash
cp .env.example .env
```

Edit `.env` and replace the placeholder values with your credentials and
MongoDB connection string. Do not commit `.env`; it is ignored by Git.

## Environment variables

| Variable | Required for | Description |
| --- | --- | --- |
| `ZENROWS_API_KEY` | URL ingestion | ZenRows API key. |
| `OPENAI_API_KEY` | URL ingestion and chat | OpenAI API key; used for vector-store operations and direct OpenAI LiteLLM calls. |
| `MONGODB_URI` | All database-backed endpoints | MongoDB connection string. |
| `MONGODB_DB` | Optional | Database name; defaults to `RAG_assessment`. |
| `MONGODB_SESSIONS_COLLECTION` | Optional | Session/vector-store collection; defaults to `sessions`. |
| `MONGODB_CHAT_COLLECTION` | Optional | Conversation collection; defaults to `chat`. |
| `OPENAI_CHAT_MODEL` | Chat | LiteLLM model or proxy model name; defaults to `gpt-5.6-sol`. |
| `LANGFUSE_PUBLIC_KEY` | Chat | Langfuse project public key. |
| `LANGFUSE_SECRET_KEY` | Chat | Langfuse project secret key. |
| `LANGFUSE_BASE_URL` | Optional | Langfuse host; defaults to `https://cloud.langfuse.com`. |
| `LITELLM_PROXY_API_KEY` | Optional | When set, route chat requests through the LiteLLM proxy instead of directly to OpenAI. |
| `LITELLM_PROXY_API_BASE` | Optional | LiteLLM proxy URL; defaults to `http://localhost:4000`. |

The `.env.example` file lists these variables with placeholder values. Do not
put real credentials in the example file.

## Run the API

Start the development server from the project root:

```bash
uv run uvicorn main:main --reload --host 0.0.0.0 --port 8000
```

The interactive API documentation is available at
[http://localhost:8000/docs](http://localhost:8000/docs). The root URL redirects
to the docs.

Run the tests with:

```bash
uv run python -m unittest discover -s tests -v
```

Runtime logs are written to `logs/response_api.log` and rotated after they reach
5 MB, keeping up to three backups.

## API endpoints

All successful responses are JSON unless otherwise noted. FastAPI's OpenAPI
schema and interactive documentation are served at `/docs`.

### `GET /`

**Use case:** Open the API documentation.

Redirects the caller to `/docs`.

### `POST /ingestion/scrape`

**Use case:** Start a website crawl and attach its Markdown pages to the
session's OpenAI vector store. Use the returned `session_id` for subsequent
chat requests.

Request body is a JSON string containing an absolute HTTP or HTTPS URL:

```json
"https://example.com"
```

Optional query parameter:

| Name | Type | Required | Description |
| --- | --- | --- | --- |
| `session_id` | string | No | Existing session ID. If omitted, a new session and vector store are created. |

Example:

```bash
curl -X POST "http://localhost:8000/ingestion/scrape" \
  -H "Content-Type: application/json" \
  -d '"https://example.com"'
```

Success response (`200`):

```json
{
  "session_id": "session-uuid",
  "vector_store_id": "vs_...",
  "pages_scraped": 5,
  "pages_uploaded": 5,
  "pages_skipped": 0
}
```

Returns `422` for an invalid URL, `503` for missing configuration or MongoDB
failure, and `502` when ZenRows or OpenAI ingestion fails.

### `POST /chat`

**Use case:** Ask a question about the content crawled for a session, or continue
an existing conversation.

Request schema:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `session_id` | string | Yes | Session ID returned by `/ingestion/scrape`. |
| `previous_interaction_id` | string or `null` | No (defaults to `null`) | Stable conversation ID returned in the previous chat response. Use `null` to start a new conversation. |
| `query` | string | Yes | User's question. |

Example:

```json
{
  "session_id": "session-uuid",
  "previous_interaction_id": null,
  "query": "What does this website say about its mission?"
}
```

On follow-up, send the response's `previous_interaction_id` with the next query:

```json
{
  "session_id": "session-uuid",
  "previous_interaction_id": "conversation-uuid",
  "query": "Can you elaborate?"
}
```

Success response (`200`):

```json
{
  "session_id": "session-uuid",
  "previous_interaction_id": "conversation-uuid",
  "output": "The answer based on the retrieved website content."
}
```

The API looks up conversation history by `session_id` and the stable
`previous_interaction_id`, retrieves relevant content from that session's
vector store, and sends both the history and retrieved content to LiteLLM. The
latest LiteLLM response ID is maintained internally in the chat document.

Returns `404` if the session or conversation is unknown, `409` if a concurrent
request advanced the conversation, `503` for missing configuration or MongoDB
failure, and `502` for provider or completion failures.

### `GET /chat/ids`

**Use case:** Populate a conversation/session list in a client.

Returns all stored session/conversation ID pairs from the `chat` collection,
ordered by creation time:

```json
[
  {
    "session_id": "session-uuid",
    "previous_interaction_id": "conversation-uuid"
  }
]
```

Returns `503` if MongoDB is unavailable or not configured.

### `GET /chat/conversation`

**Use case:** Load one conversation's full history, for example when a user
selects it from a conversation list.

Required query parameters:

| Name | Type | Description |
| --- | --- | --- |
| `session_id` | string | Session containing the conversation. |
| `previous_interaction_id` | string | Stable conversation ID returned by `/chat`. |

Example:

```bash
curl --get "http://localhost:8000/chat/conversation" \
  --data-urlencode "session_id=session-uuid" \
  --data-urlencode "previous_interaction_id=conversation-uuid"
```

Success response (`200`):

```json
{
  "session_id": "session-uuid",
  "previous_interaction_id": "conversation-uuid",
  "interactions": [
    {
      "user": "What does this website say about its mission?",
      "system": "The website describes its mission as..."
    }
  ],
  "created_at": "2026-10-16T10:00:00Z"
}
```

Returns `404` if that exact session/conversation pair does not exist, or `503`
if MongoDB is unavailable or not configured.

## Data and CORS notes

- MongoDB's `sessions` collection maps session IDs to OpenAI vector-store IDs.
- MongoDB's `chat` collection stores one conversation document per
  `(session_id, previous_interaction_id)` conversation, with turns in its
  `interactions` array.
- Scraped page Markdown is uploaded to OpenAI; it is not stored locally or in
  MongoDB.
- CORS currently allows any origin, method, and header. Restrict allowed origins
  before exposing the API in production.
