import { spawnSync } from 'node:child_process';
import { chmodSync, mkdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { join } from 'node:path';

process.chdir(fileURLToPath(new URL('..', import.meta.url)));
process.umask(0o077);
mkdirSync('backups', { recursive: true, mode: 0o700 });
const output = join('backups', `brittain-${new Date().toISOString().replace(/[:.]/g, '-')}.sql`);
const exported = spawnSync('npx', ['--no-install', 'wrangler', 'd1', 'export', 'DB', '--remote', '--output', output], { stdio: 'inherit' });
if (exported.status !== 0) process.exit(exported.status || 1);
chmodSync(output, 0o600);
if (!statSync(output).size) throw new Error('The database export is empty.');
// Restore into a temporary in-memory database. Never run a restore against production here.
const verified = spawnSync('python3', ['-c', `
import sqlite3, sys
from pathlib import Path
db = sqlite3.connect(':memory:')
db.executescript(Path(sys.argv[1]).read_text())
assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Integrity check failed'
assert not db.execute('PRAGMA foreign_key_check').fetchall(), 'Foreign key check failed'
required = {'user', 'session', 'account', 'verification', 'chats', 'chat_exchanges'}
tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
assert required <= tables, 'Required tables are missing'
print('Restore check passed: database integrity, foreign keys, and required tables.')
`, output], { stdio: 'inherit' });
if (verified.status !== 0) process.exit(verified.status || 1);
console.log(`Verified private backup: ${output}`);
