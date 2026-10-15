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