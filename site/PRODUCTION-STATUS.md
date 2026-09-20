# Production preparation

Updated 2026-09-19 (America/Chicago).

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

## External setup still needed

- Resend: create or select the account, verify the sending domain, and supply `RESEND_API_KEY`. The current sending address is `accounts@brittain.app`. Configure the DNS records supplied by Resend and test actual delivery with an authorized recipient. No emails were sent by this preparation work.
- Model host: access to the WSL machine is needed to install and supervise Cloudflare Tunnel and the model process. Keep the existing ngrok origin until the replacement works.
- Policies: select a support/privacy mailbox, retention periods, and final terms. No contact address or retention promise has been invented.
- Domain: `brittain.app` still needs the planned cutover from Sites after the account and model checks pass. This update does not change its binding.
- Staging: a separate Worker/database and independent test secrets still need setup.
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
