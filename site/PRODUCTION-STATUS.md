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
- Replaced obsolete GitHub Pages deployment with tests, lint, dependency audit, and Cloudflare builds. The [GitHub check for the staging commit](https://github.com/LukeMcB1128/brittain-model/actions/runs/35520840151) passed.
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

The user approved the production export on 2026-09-20. The backup was saved under ignored `site/backups/` and restored into an in-memory SQLite database. Database integrity, foreign keys, and all six required tables passed. No production data was restored or changed.

- Backup file: `backups/brittain-2026-09-20T15-51-44-459Z.sql`, 1,214,929 bytes.
- Permissions checked: directory `0700`, file `0600`. Git ignores the backup.
- Recovery bookmark recorded after export: `0000002e-00000000-000050ec-2c64a6fb72c2abea94ed8283374c3a65`.
- The backup command now hides signed download URLs in both success and error output. Two tests verify URL filtering and private file permissions with synthetic data.
- This is a local backup with a SQLite restore check. Remote D1 restore, encrypted off-device storage, and scheduled backups have not been tested or configured.

## Verified deployment

- Active Worker version: `42355249-f5aa-4166-bfb9-3250ce1bfac1`. It includes the Brave runtime fix, the bounded web-evidence final-answer flow, safe currency rendering, keyboard navigation fixes, removal of the obsolete experimental chat route, and the hosted-only Brittain 4 release policy.
- Previous known-good Worker version: `11d15be2-fa92-455e-8142-0fcdf9869f29`. It includes the same chat, tool, accessibility, and route fixes but still describes planned Brittain 4 downloads.
- Previous release before this update: `63e9c12b-8b54-4b17-8dfe-950233a73e02`. Rolling back to it also reverts the new account protection; assess that tradeoff before rollback.
- Tests: 114 passing. Lint and the production build pass. The build warns that production secrets are absent locally; live health checks confirm the deployed configuration is present.
- Live release check: 19 of 20 checks pass on the Workers address. Email verification/password recovery is the remaining failed check.
- Live negative login check: HTTP 400 with `MISSING_RESPONSE` when no CAPTCHA token is supplied.
- Browser check: the managed widget completed automatically. A deliberately invalid login returned the expected error, refreshed the token, and enabled another attempt. No account was created and no email was sent.
- Static pages return the security headers. Chat URLs return `noindex`, including direct conversation links.
- Responsive checks at 320 px and 390 px cover the home, account, and model pages. No horizontal overflow was found.
- The Models tabs support Left Arrow, Right Arrow, Home, and End. A live browser check confirmed that Right Arrow selects and focuses Experimental models. The production page reported no browser warnings or errors.
- Public pages include a keyboard skip link. A live browser check confirmed that it moves focus past the header to the start of page content.
- The chat sidebar now reports its mobile open state correctly. Escape, the close button, and the scrim return focus to the open-sidebar button.
- Currency amounts no longer become accidental LaTeX spans. Inline and display equations continue to use KaTeX.
- The obsolete `/experimental-chat` route and its direct browser-to-ngrok client are no longer in the production bundle. A live browser check confirmed that the route returns the standard Page not found screen.
- Brittain 4 is now described as web-chat only across the home, account, Models, and model-detail pages. Experimental model downloads remain separate. A live browser check confirmed both model views and the direct Experimental models link.

## Dependency audit

The four moderate findings were removed with a scoped override: `@esbuild-kit/core-utils` uses the project's existing `esbuild` version, currently `0.28.2`. Other locked package versions are unchanged. The [esbuild advisory](https://github.com/advisories/GHSA-67mh-4wv8-2f99) covers versions through `0.24.2`.

`npm audit` now reports zero vulnerabilities. Database tooling loaded the TypeScript schema, generated all six tables, and passed a SQLite integrity check on the generated SQL. All 105 tests, lint, and both environment builds pass. CI now fails on moderate or higher findings. These tooling and backup changes do not need a Worker deployment.

## Staging verification

- Site: https://brittain-app-staging.luke-brittain.workers.dev.
- Worker version: `3e150745-ad15-4a3a-ab89-8689c478d812`.
- Database: `brittain-app-staging` (`f0376451-2abf-47e9-b938-dfb007402221`). Both schema migrations applied.
- All 103 tests pass. Lint, staging build, and production build pass. The production build was checked for the correct Worker, database, and enabled chat setting.
- `npm run check:staging`: 22 checks pass; email readiness is explicitly skipped.
- Live requests confirm that chat generation returns maintenance status and login without CAPTCHA is rejected.
- Browser shows the test-site label and a successful managed CAPTCHA. No account was created or email sent.
- The health check allows an absent model key only while chat is paused. This change is now deployed to both environments.

## Search failure recovery

A supplied chat record showed four failed searches, six HTTP 404 page reads, and repeated promises to write code. Search was removed from the offered tools after the first failure, but the server still executed it when the model requested it again.

- The server now rejects withdrawn or unoffered tools before execution.
- Three consecutive failures end tool use and request a final answer from the conversation. Successful calls reset the failure count.
- The 15-call limit is enforced within each batch, not only between model rounds.
- A model that still requests tools in the final round gets an explicit error; the reply is not marked complete.
- Failed searches include safe provider status codes. Response bodies, keys, and raw network errors are excluded.
- Six regression tests cover these cases. Both environments are deployed and staging checks pass.
- The production Brave secret exists. A later record reported a Brave request failure and a DuckDuckGo timeout. The cause was `redirect: 'error'`: Cloudflare's runtime rejects that option before sending the Brave request. A local workerd check reproduced the exact error.
- The request now uses `redirect: 'manual'`. Redirect responses are rejected, so the key cannot be forwarded to another host. A regression test uses the real Cloudflare Request constructor with synthetic provider responses; all 112 tests pass.
- A live signed-in search with the production provider key still needs verification; the available test browser is signed out.

## Remaining launch checks

- [ ] Complete email verification and password recovery on the final domain.
- [x] Verify Turnstile in a real browser and recover after a wrong password.
- [ ] Load-test the model with four concurrent requests and tool calls.
- [ ] Test tab close, disconnection, Stop, compaction, and reload with a signed-in test account.
- [x] Approve and execute the local database backup/restore check.
- [ ] Configure uptime/error/capacity alerts.
- [ ] Complete a screen-reader check. Mobile widths and the main keyboard interactions have been checked.
- [ ] Run `npm run check:release -- https://brittain.app` after cutover.
- [x] Record a known-good Worker version and D1 bookmark for rollback.
