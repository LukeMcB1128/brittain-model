import test from 'node:test';
import assert from 'node:assert/strict';
import { chmodSync, copyFileSync, mkdirSync, mkdtempSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

for (const fails of [false, true]) {
  test(`backup hides signed URLs when export ${fails ? 'fails' : 'succeeds'}`, () => {
    const root = mkdtempSync(join(tmpdir(), 'brittain-backup-test-'));
    try {
      mkdirSync(join(root, 'scripts'));
      mkdirSync(join(root, 'bin'));
      mkdirSync(join(root, 'backups'));
      chmodSync(join(root, 'backups'), 0o755);
      copyFileSync(new URL('./backup-db.mjs', import.meta.url), join(root, 'scripts/backup-db.mjs'));
      // Synthetic data only. The fake CLI never contacts Cloudflare.
      writeFileSync(join(root, 'bin/npx'), `#!/usr/bin/env node
const fs = require('node:fs');
console.log('Download: https://example.invalid/export?signature=private-test-token');
console.error('https://example.invalid/export?signature=private-test-token');
if (${fails}) process.exit(2);
const output = process.argv[process.argv.indexOf('--output') + 1];
fs.writeFileSync(output, ['user', 'session', 'account', 'verification', 'chats', 'chat_exchanges'].map(t => 'CREATE TABLE "' + t + '" (id TEXT PRIMARY KEY);').join(''));
`, { mode: 0o700 });
      const result = spawnSync(process.execPath, [join(root, 'scripts/backup-db.mjs')], {
        encoding: 'utf8', env: { ...process.env, PATH: `${join(root, 'bin')}:${process.env.PATH}` },
      });
      assert.equal(result.status, fails ? 2 : 0, result.stderr);
      assert.doesNotMatch(result.stdout + result.stderr, /private-test-token|https:\/\/example/);
      assert.match(result.stdout, /\[private URL omitted\]/);
      assert.equal(statSync(join(root, 'backups')).mode & 0o777, 0o700);
      if (!fails) {
        assert.match(result.stdout, /Restore check passed/);
        const files = readdirSync(join(root, 'backups'));
        assert.equal(files.length, 1);
        assert.equal(statSync(join(root, 'backups', files[0])).mode & 0o777, 0o600);
      }
    } finally { rmSync(root, { recursive: true, force: true }); }
  });
}
