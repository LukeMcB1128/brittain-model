# Standalone deployment

The production site is a React application and Cloudflare Worker in one build.
The Worker owns account sessions, chat history, model API access, and response
headers. A visitor does not need a ChatGPT account.

## Local setup

```sh
cd site
cp .dev.vars.example .dev.vars
# Add local secret values to .dev.vars.
npm ci
npm run db:migrate:local
npm run dev -- --host 127.0.0.1
```

Wrangler keeps the local D1 data under `site/.wrangler/`. This directory and
`.dev.vars` are ignored by Git.

## Cloudflare resources

1. Create a D1 database named `brittain-app`.
2. Put its database ID in `wrangler.jsonc`.
3. Apply the migrations with `npm run db:migrate:remote`.
4. Set `BETTER_AUTH_SECRET` with at least 32 random characters.
5. Set `BRITTAIN4_API_KEY` as a Worker secret.
6. Set `MODEL_API_ORIGIN` to the stable Brittain model origin.
7. Set `RESEND_API_KEY` to require email verification and enable password reset.
8. Set `TURNSTILE_SECRET_KEY` and `TURNSTILE_SITE_KEY` to protect account forms.
9. Set `CHAT_MAX_CONCURRENT` from model-server load tests. It defaults to four.
10. Set `CHAT_MAX_PER_HOUR` to the measured service limit. It defaults to 621.
11. `CHAT_LOCK_TIMEOUT_SECONDS` limits stale response locks. It defaults to 210 seconds.

Use `wrangler secret put NAME` for secret values. Do not add secrets to
`wrangler.jsonc` or to a `VITE_` variable.

## Staging and release

Deploy and test a staging Worker before the domain change. Check these flows:

- Create an account, verify the email, sign in, and sign out.
- Save, reload, and delete a conversation.
- Stream a response and stop it.
- Use web, calculator, PDF, and image features.
- Compact a long conversation and reload it.
- Reject an unauthenticated `/api/chat` request.
- Show a clear error when the model server is unavailable.

After staging passes, add `brittain.app` as the Worker Custom Domain. Remove the
old ChatGPT Sites domain binding only during that cutover. Keep the old
`chatgpt.site` address for rollback until the standalone site is stable.

## Stored data

D1 stores accounts, sessions, saved conversation text, compacted memory, tool
summaries, and redacted exchange records. It does not store image or PDF binary
data in saved chat payloads. Attached PDF text can remain in a saved chat so the
conversation stays understandable after reload.
