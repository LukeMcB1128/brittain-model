# Brittain 4 website

The main chat uses a standalone Cloudflare Worker. Users create a Brittain
account with email and password. Better Auth stores sessions in D1, and saved
chats follow the account across visits. The Worker calls the authenticated
Brittain 4 `/v1/chat/completions` endpoint. The key is never included in browser
files.

See [`STANDALONE.md`](STANDALONE.md) for local setup, Cloudflare resources,
staging, and domain cutover.

## Local development

Copy `.dev.vars.example` to `.dev.vars`, add local secret values, apply the local
D1 migrations, and run `npm run dev -- --host 127.0.0.1`.

## Previous Sites build

`npm run build:hosted` remains only for the existing ChatGPT Sites rollback
deployment. New releases use `npm run build` and `npm run deploy`.

Do not publish a secret using a `VITE_` variable.

## Conversation compaction

The Brittain 4 chat keeps recent turns unchanged. When the active conversation grows past the safe request budget, the gateway asks Brittain 4 to summarize the oldest complete turns. The page keeps that memory with the chat and sends it with later requests. Old attachment bytes are removed from the active request after their turns are compacted. If the summary request fails, the gateway uses the full conversation and reports the failure in the current turn.

## Transcript records

Nothing recorded what the API served. vLLM logs one status line per request with
no message content, and the gateway streamed replies to the browser and forgot
them, so hundreds of real exchanges were unrecoverable. Real usage is the most
useful source of training and eval data this project has.

Recording is **off unless a binding is present**, and the gateway behaves exactly
as before when none is:

| binding | effect |
| --- | --- |
| `DB` | the Sites D1 database; one `chat_exchanges` row per exchange |
| `CHAT_LOG` | a KV namespace; one key per exchange, `chat/<iso-timestamp>/<uuid>`, so a listing comes back in order |
| `CHAT_LOG_URL` | an endpoint that receives the entry as a JSON POST |
| `CHAT_LOG_TOKEN` | optional bearer token sent with `CHAT_LOG_URL` |

An entry holds the conversation, the reply, tool calls with their arguments and
outcomes, token usage, finish reason, duration, and any error. Failed exchanges
are recorded too — they are the more informative ones.

### Apply the migration before expecting rows

`npm run db:generate` writes a migration from `db/schema.ts`; applying it to the
D1 database is a separate step, done with whatever Sites provides for running
SQL against the binding. Until `chat_exchanges` exists, every insert raises
`no such table`, and because a lost record must never become a lost reply that
error is swallowed — so the symptom is not an error anywhere, it is zero rows.
Verify after deploying rather than assuming:

```sql
SELECT count(*) AS exchanges, max(created_at) AS latest FROM chat_exchanges;
```

If that returns rows, recording works. If it errors, the migration has not been
applied. If it returns zero with a table present, nothing has been served since
the binding was added.

What it deliberately does not hold:

- **Attachment bytes.** Images and PDFs are counted, never stored. A base64
  payload is the bulk of a request and worthless as a record.
- **Who sent it.** The authenticated user id is stored as a truncated SHA-256
  digest, which groups a person's exchanges without recording their identity.
- **Credentials.** API keys, tokens and password assignments are redacted before
  anything is written. The existing chat corpus contained plaintext credentials,
  and a record that preserves them outlives the chat that leaked them.
- **The system prompt.** It is identical on every request and lives in source.

Writing happens after the reply has been delivered to the client, and every
failure is swallowed: a lost record must never become a lost reply.

If the site ever takes visitors beyond people you have told, this needs a line in
the UI saying conversations are kept, and a retention limit. Storing strangers'
conversations indefinitely is a different thing from keeping your own.

## Chat behaviour

- Replies stream over SSE. Thinking is off.
- Stop cancels the request and keeps partial text. Retry replaces the last reply.
- Conversations are saved to the signed-in account in D1.
- History is sent without silent trimming. Replies are capped at 2,048 output tokens. The upstream server rejects a request if prompt plus output exceeds its 32,768-token context.
- Context and service errors remain visible. The tunnel's `/tokenize` route returned HTTP 502 during integration, so exact preflight token counting is not enabled.
- Search, calculator, PDF, image, and text-file controls are enabled.

Run `npm test`, `npm run lint`, and `npm run build` to verify the integration.

## Earlier experimental chat

The retained experimental interface is at `/experimental-chat`. It uses the old Ollama-compatible API, which differs from the Brittain 4 API. The following notes apply only to that interface.

### Experimental chat

React + Vite frontend for the Ollama-compatible server in `scripts/inference/serve.py`.

## Run locally

```sh
cd site
npm ci
npm run dev
```

Open the URL Vite prints (normally `http://localhost:5173/brittain-model/`).
The default API origin is the existing ngrok tunnel. To change it, put
`VITE_API_ORIGIN=https://your-tunnel.ngrok-free.dev` in `site/.env.local`, then
restart Vite. Production builds embed this value at build time.

## Conversations

- Instruct models use `/api/chat` with prior user and assistant messages.
- Restart the updated `serve.py` on the GPU host to enable history. The frontend
  detects older servers and clearly labels their single-turn behavior.
- Shakespeare uses its native role markers. The browser hides its entire tag
  block and all control markers, including partially streamed markers. Copying
  a reply copies only the visible text. Private tag blocks stay in context.
- Coder instruct uses the Alpaca Input field for earlier conversation and the
  Instruction field for the newest request. Supplying history does not add
  multi-turn training to a checkpoint; follow-up quality remains model-dependent.
- Raw models use `/api/generate` with the latest code prefix unchanged, including
  trailing indentation. Earlier turns remain visible but are not sent as code.
- The server reserves output tokens and drops the oldest complete turns until
  context fits. Oversized latest messages receive an actionable error instead
  of being silently cut. The UI reports when earlier messages were omitted.
- Chats are saved in this browser's local storage. They do not sync across
  devices. Selecting another model starts a separate conversation.
- Enter sends; Shift+Enter inserts a newline. Stop cancels the fetch and keeps
  partial output. Try again replaces the latest reply without duplicating the
  user turn. Fenced code and basic inline Markdown are rendered without HTML.

## Verify

```sh
npm test
npm run build
npm run lint
# From the repository root (no GPU or third-party Python packages needed):
python -m unittest discover -s tests -p test_chat_context.py -v
```

## Publish

The existing `.github/workflows/pages.yml` builds and publishes changes under
`site/` when pushed to `main`. GitHub Pages stays at
`https://lukemcb1128.github.io/brittain-model/`. Update/restart the GPU server
alongside the frontend deployment; GitHub Pages does not deploy Python code.
