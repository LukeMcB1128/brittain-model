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

Use `wrangler secret put NAME --config wrangler.jsonc --env ""` for production secret values. Do not add secrets to
`wrangler.jsonc` or to a `VITE_` variable.

## Web search

Set `BRAVE_SEARCH_API_KEY` as a Worker secret with
`npx wrangler secret put BRAVE_SEARCH_API_KEY --config wrangler.jsonc --env ""`. For local development, add it
to `.dev.vars`. Get the key from <https://api-dashboard.search.brave.com/>.
The server uses the Brave Search API when this secret is set. The key is never
sent to the browser or model.

If the API fails, the server tries DuckDuckGo HTML once. Without the secret,
only DuckDuckGo is used. This fallback can return bot challenges, especially
from hosted servers, so it is not a reliable production search provider.
Each provider has an eight-second timeout. Stopping the reply cancels search.
If search is unavailable, the model is told not to claim it verified current
facts. A valid empty result set does not disable search for other queries.

## Staging and release

The staging site is https://brittain-app-staging.luke-brittain.workers.dev.
It has a separate D1 database, account secret, Turnstile widget, rate limits,
and Durable Object namespace. Production accounts and chats are not copied.
The page title and public header identify it as a test site. All staging pages
have a `noindex` header.

Run these commands from `site/`:

```sh
npm run db:migrate:staging
npm run deploy:staging
npm run check:staging
```

Staging starts with `CHAT_ENABLED=false`. It has no model or email credentials.
The staging check verifies this paused state and skips email readiness. It is
not a production release check. To test model responses later, set a test model
origin and add its key to the staging Worker, then enable chat in `env.staging`.
Use test data only. Complete the email service setup before testing delivery.

Use `npx wrangler secret put NAME --config wrangler.jsonc --env staging` for
staging secrets. For production, use `--env ""`. Use `npm run deploy` for
production. Both deploy commands build the selected environment first.
Do not run plain `wrangler deploy` after a staging build: Vite records the last
build as the deployment target. Database commands select their environment
explicitly so the last build does not change the database they use.

Before the domain change, check these flows on staging after its model and
email services are configured:

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

## Release and recovery commands

Run these from `site/`:

```sh
npm test
npm run lint
npm run build
npm run check:release -- https://brittain-app.luke-brittain.workers.dev
```

Release checks are read-only. They report a failure while email or Turnstile is not configured. They do not test real email delivery or model inference.
See `PRODUCTION-STATUS.md` for the remaining launch work.

The account forms use two Turnstile keys. Upload both together with `wrangler secret bulk` from a private JSON file. Never put the secret in a `VITE_` variable. The widget must allow the exact hostnames used for signup and login. Do not add localhost to the production widget; use Cloudflare test keys for local testing.

The auth limits use the Cloudflare client IP header. They are per-location burst protection, not a global daily quota. The chat capacity gate is the separate global limit.

To pause new replies, set `CHAT_ENABLED` to `"false"` in `wrangler.jsonc` and deploy. Set it back to `"true"` to resume. This does not cancel active replies or prevent users from reading saved chats.

`GET /api/health` checks the account secret and database. When chat is enabled, it also requires the model key. A paused site can return 200 without a model key and reports `chatPaused: true`. The authenticated `/api/session` check verifies the model connection. Neither endpoint sends a model generation request.

Before a release, record recovery information:

```sh
npx wrangler deployments list --config wrangler.jsonc --env ""
npx wrangler d1 time-travel info DB --json --config wrangler.jsonc --env ""
```

After approval to copy account/chat data to this Mac:

```sh
npm run db:backup
```

This exports D1 to `backups/` with private file permissions and restores the SQL into an in-memory SQLite database for integrity and foreign-key checks. Keep exports out of Git and public storage. Move retained backups to encrypted storage under the owner's control. The script requires Python 3.

For an application-only rollback, use `npx wrangler rollback <known-good-version-id> --config wrangler.jsonc --env ""` and rerun the release checks. A database restore is separate and can discard newer writes. Pause chat, preserve the current database, and obtain explicit approval before using `wrangler d1 time-travel restore` with the recorded bookmark. Check the account's available recovery window first.

References:
- https://developers.cloudflare.com/turnstile/get-started/
- https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/
- https://developers.cloudflare.com/d1/reference/time-travel/
- https://developers.cloudflare.com/workers/vite-plugin/reference/cloudflare-environments/
