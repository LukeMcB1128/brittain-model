# Production preparation

Updated 2026-09-20 (America/Chicago).

## Completed in this update

- Created and enabled a managed Turnstile widget for `brittain.app` and `brittain-app.luke-brittain.workers.dev`. Both keys are stored as Worker secrets.
- Added account request limits before password and CAPTCHA processing: 20 POST requests per minute per client IP. Signup and email requests share a stricter limit of 3 per minute. These Cloudflare counters are per location, not exact global quotas.
- Added recovery for expired or consumed CAPTCHA tokens, failed widget loading, and failed account configuration loading.
- Fixed password reset: it no longer requires a token from an absent CAPTCHA widget.
- Reject incomplete Turnstile configuration and validate widget hostnames on the server.
- Reject external post-login redirects, including backslash forms.
- Reserve chat capacity before database work. A completed old request cannot release the slot of a newer request.
- Added `CHAT_ENABLED=false` to pause new generations while accounts and saved chats remain available.
- Added `/api/health`. It checks Worker configuration and D1. It does not check model inference.
- Added security headers for static pages. The existing Worker headers only covered API responses.
- Added indexing exclusions for account/chat routes, a robots file, and a sitemap.
- Replaced obsolete GitHub Pages deployment with tests, lint, dependency audit, and a Cloudflare build. This workflow will run after the commit is pushed to GitHub.
- Added read-only release checks and a private database backup/restore-check command.
- Committed the deployed production preparation as `6cb3ef7`.
- Created and deployed a separate staging Worker, empty D1 database, Turnstile widget, and account secret. No production data or credentials were copied.
- Added separate staging build, deployment, migration, and check commands. Production database commands use an explicit target.

## External setup still needed

- Resend: create or select the account, verify the sending domain, and supply `RESEND_API_KEY`. The current sending address is `accounts@brittain.app`. Configure the DNS records supplied by Resend and test actual delivery with an authorized recipient. No emails were sent by this preparation work.
- Model host: access to the WSL machine is needed to install and supervise Cloudflare Tunnel and the model process. Keep the existing ngrok origin until the replacement works.
- Policies: select a support/privacy mailbox, retention periods, and final terms. No contact address or retention promise has been invented.
- Domain: `brittain.app` still needs the planned cutover from Sites after the account and model checks pass. This update does not change its binding.
- Staging model/email tests: the separate site is live, but chat is paused and model/email credentials are absent. Add test service configuration before testing those flows.
- Release assets: model files, license, hardware requirements, and evaluation results.
- Monitoring: health endpoint is available, but external uptime checks and alert recipients are not configured.

## Recovery status

Cloudflare returned a D1 recovery bookmark during preparation. No production data was restored or changed.
The attempted full backup was blocked by automatic approval review because it would copy production account/chat data to this Mac. Explicit approval is needed to run `npm run db:backup`.
The backup command is implemented; its full-data restore check has not been run. It writes a private SQL export under ignored `site/backups/`, then verifies it in an in-memory SQLite database.

## Verified deployment

- Active Worker version after the Turnstile secret update: `1fb032e3-9219-4321-a5cc-f2c0e3a8d7c8`.
- Previous release before this update: `63e9c12b-8b54-4b17-8dfe-950233a73e02`. Rolling back to it also reverts the new account protection; assess that tradeoff before rollback.
- Tests: 102 passing. Lint and production build pass. The build warns that production secrets are absent locally; live health checks confirm the deployed configuration is present.
- Live release check: 19 of 20 checks pass on the Workers address. Email verification/password recovery is the remaining failed check.
- Live negative login check: HTTP 400 with `MISSING_RESPONSE` when no CAPTCHA token is supplied.
- Browser check: the managed widget completed automatically. A deliberately invalid login returned the expected error, refreshed the token, and enabled another attempt. No account was created and no email was sent.
- Static pages return the security headers. Chat URLs return `noindex`, including direct conversation links.

## Dependency audit

The audit found four moderate findings in the transitive `drizzle-kit` / `@esbuild-kit` / older `esbuild` chain, and no high or critical findings. These concern development-server tooling. Do not expose the database tooling server publicly. A breaking downgrade suggested by npm was not applied. CI fails on high or critical findings; the moderate tooling update remains follow-up work.

## Staging verification

- Site: https://brittain-app-staging.luke-brittain.workers.dev.
- Worker version: `13300e38-78d1-4db5-842b-29701079cc29`.
- Database: `brittain-app-staging` (`f0376451-2abf-47e9-b938-dfb007402221`). Both schema migrations applied.
- All 103 tests pass. Lint, staging build, and production build pass. The production build was checked for the correct Worker, database, and enabled chat setting.
- `npm run check:staging`: 22 checks pass; email readiness is explicitly skipped.
- Live requests confirm that chat generation returns maintenance status and login without CAPTCHA is rejected.
- Browser shows the test-site label and a successful managed CAPTCHA. No account was created or email sent.
- The staging health check allows an absent model key only while chat is paused. This change is deployed on staging; production still uses the version listed above.

## Remaining launch checks

- [ ] Complete email verification and password recovery on the final domain.
- [x] Verify Turnstile in a real browser and recover after a wrong password.
- [ ] Load-test the model with four concurrent requests and tool calls.
- [ ] Test tab close, disconnection, Stop, compaction, and reload with a signed-in test account.
- [ ] Approve and execute the database backup/restore check.
- [ ] Configure uptime/error/capacity alerts.
- [ ] Check mobile browsers, keyboard use, and screen readers.
- [ ] Run `npm run check:release -- https://brittain.app` after cutover.
- [ ] Keep a known-good Worker version and D1 bookmark for rollback.
