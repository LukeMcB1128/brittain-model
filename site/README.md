# BRITTAIN chat

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
