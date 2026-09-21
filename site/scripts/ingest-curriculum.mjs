// Load the AISD/TEA curriculum into D1, one row per course.
//
// WHY THIS IS RETRIEVAL AND NOT TRAINING DATA
// TEKS codes, PEIMS numbers and course numbers have to be exact, and they are
// revised yearly. A model that has memorised them gives you no way to tell a
// remembered code from an invented one, and a wrong TEKS citation reads exactly
// like a right one. So the curriculum lives in a table the model queries, and
// answers quote it.
//
// WHY D1 AND NOT THE MODEL TUNNEL
// The curriculum is useful whether or not the GPU machine is awake. Serving it
// from the same host as vLLM would couple every lookup to that machine being
// on, which it frequently is not.
//
// WHAT IS NOT INGESTED
// sources/ holds the catalog PDFs and TEKS chapter dumps -- 19 MB of provenance.
// Answers should cite those paths, not carry their contents. Only the 394
// curated course files go in.
//
//   node scripts/ingest-curriculum.mjs --source ../../aisd-curriculum --out /tmp/courses.sql
//   npx wrangler d1 execute brittain-app --remote --file /tmp/courses.sql
import { readdir, readFile, stat, writeFile } from 'node:fs/promises';
import { join, posix } from 'node:path';

const args = new Map();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i], process.argv[i + 1]);
const SOURCE = args.get('--source') || '../aisd-curriculum';
const OUT = args.get('--out') || 'curriculum.sql';

// Directories that are not subjects.
const SKIP = new Set(['.git', 'sources', 'node_modules']);

// Each file opens with a `# Title` and a Course Overview block of bullets.
// Anything missing is left null rather than guessed: a null PEIMS code is
// honest, an invented one is not.
function field(body, label) {
  const match = body.match(new RegExp(`^- \\*\\*${label}:\\*\\*\\s*(.+)$`, 'm'));
  if (!match) return null;
  // Several fields run the prerequisite and the description together in the
  // source. Keep the first sentence-ish span; the full text stays in `body`.
  return match[1].trim().slice(0, 300) || null;
}

function parse(slug, subject, body) {
  const title = body.match(/^#\s+(.+)$/m)?.[1]?.trim();
  if (!title) return null;
  const teks = body.match(/\((19 TAC [^)]+?)(?: for [^)]+)?\)/)?.[1]?.trim() || null;
  return {
    slug,
    subject,
    title,
    credit: field(body, 'Credit'),
    gradeLevel: field(body, 'Grade Level'),
    courseNumber: field(body, 'Course Number'),
    peims: field(body, 'PEIMS Code')?.replace(/^#/, '') || null,
    teksCite: teks,
    body,
  };
}

const quote = value => (value === null || value === undefined
  ? 'NULL'
  : `'${String(value).replace(/'/g, "''")}'`);

const rows = [];
const problems = [];

for (const subject of (await readdir(SOURCE, { withFileTypes: true }))
  .filter(entry => entry.isDirectory() && !SKIP.has(entry.name))
  .map(entry => entry.name)
  .sort()) {
  for (const file of (await readdir(join(SOURCE, subject))).sort()) {
    if (!file.endsWith('.md') || file.startsWith('.')) continue;
    const path = join(SOURCE, subject, file);
    const body = await readFile(path, 'utf8');
    const slug = posix.join(subject, file.replace(/\.md$/, ''));
    const row = parse(slug, subject, body);
    if (!row) { problems.push(`${slug}: no title heading`); continue; }
    row.updatedAt = (await stat(path)).mtime.toISOString();
    rows.push(row);
  }
}

if (!rows.length) throw new Error(`no course files under ${SOURCE}`);

// INSERT OR REPLACE keyed on slug, so re-running after a curriculum update is
// the whole refresh procedure. Courses deleted upstream are removed first.
const statements = [
  'DELETE FROM courses;',
  ...rows.map(row => 'INSERT INTO courses (slug, subject, title, credit, grade_level, '
    + 'course_number, peims, teks_cite, body, updated_at) VALUES ('
    + [row.slug, row.subject, row.title, row.credit, row.gradeLevel, row.courseNumber,
      row.peims, row.teksCite, row.body, row.updatedAt].map(quote).join(', ')
    + ');'),
];

await writeFile(OUT, statements.join('\n') + '\n', 'utf8');

const bySubject = rows.reduce((acc, row) => {
  acc[row.subject] = (acc[row.subject] || 0) + 1;
  return acc;
}, {});
const bytes = rows.reduce((sum, row) => sum + row.body.length, 0);

console.log(`${rows.length} courses -> ${OUT}  (${(bytes / 1048576).toFixed(2)} MB of markdown)`);
for (const [subject, count] of Object.entries(bySubject).sort()) {
  console.log(`  ${subject.padEnd(16)} ${count}`);
}
// Fields that failed to parse are reported rather than silently null, because
// a column that is empty everywhere means the source format changed.
for (const column of ['credit', 'gradeLevel', 'courseNumber', 'peims', 'teksCite']) {
  const missing = rows.filter(row => !row[column]).length;
  if (missing) console.log(`  ${String(missing).padStart(4)} courses missing ${column}`);
}
for (const problem of problems) console.log(`  SKIPPED ${problem}`);
