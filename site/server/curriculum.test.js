import test from 'node:test';
import assert from 'node:assert/strict';
import { CURRICULUM_TOOL, searchCurriculum } from './curriculum.js';
import { executeTool } from './tools.js';

// A course file as the ingest script stores it: metadata columns parsed out of
// the overview block, the whole markdown in `body`.
const CALC = {
  slug: 'math/ap-calculus-ab',
  subject: 'math',
  title: 'AP Calculus AB',
  credit: '1 (Mathematics)',
  grade_level: '11-12',
  course_number: '3614.P000.Y',
  peims: 'A3100101',
  teks_cite: '19 TAC Chapter 111',
  body: '# AP Calculus AB\n\n## Units of Study\n\n### Unit 1: Limits and Continuity\n'
    + 'Defining limits, estimating limits from graphs and tables, asymptotes, '
    + 'continuity, Intermediate Value Theorem.\n',
};
const STATS = { ...CALC, slug: 'math/ap-statistics', title: 'AP Statistics', peims: 'A3100200', body: '# AP Statistics\n\nSampling and experimentation.\n' };

// Minimal D1: prepare().bind().all(). Records the SQL and binds so the tests
// can assert what was asked for, not only what came back.
function db(rows, calls = []) {
  return {
    calls,
    prepare(sql) {
      const call = { sql, binds: null };
      calls.push(call);
      return {
        bind(...binds) { call.binds = binds; return this; },
        async all() { return { results: typeof rows === 'function' ? rows(call) : rows }; },
      };
    },
  };
}

test('the tool is declared with the arguments the model needs', () => {
  assert.equal(CURRICULUM_TOOL.function.name, 'search_curriculum');
  assert.deepEqual(
    Object.keys(CURRICULUM_TOOL.function.parameters.properties).sort(),
    ['course', 'full', 'query', 'subject'],
  );
});

test('a named course comes back whole, with its codes', async () => {
  const result = await searchCurriculum({ course: 'AP Calculus AB' }, { db: db([CALC]) });
  assert.equal(result.error, undefined);
  // The codes are the part that must be exact, so they are surfaced as fields
  // rather than left for the model to find in the prose.
  assert.match(result.content, /3614\.P000\.Y/);
  assert.match(result.content, /A3100101/);
  assert.match(result.content, /19 TAC Chapter 111/);
  assert.match(result.content, /Intermediate Value Theorem/);
  assert.equal(result.display.detail, 'AP Calculus AB');
});

test('curriculum is presented as authoritative, unlike web content', async () => {
  const result = await searchCurriculum({ course: 'AP Calculus AB' }, { db: db([CALC]) });
  // web_search and web_fetch carry a notice telling the model to treat their
  // output as untrusted evidence. This is the district's own curriculum and
  // the wording is the point: paraphrasing a TEKS expectation is the failure.
  assert.doesNotMatch(result.content, /untrusted external web content/i);
  assert.match(result.content, /verbatim/i);
});

test('a subject lists courses rather than returning every file', async () => {
  const calls = [];
  const result = await searchCurriculum({ subject: 'math' }, { db: db([CALC, STATS], calls) });
  assert.match(result.content, /AP Calculus AB/);
  assert.match(result.content, /AP Statistics/);
  // 178 CTE course files would fill the context and answer nothing, so a browse
  // must not select bodies at all.
  assert.doesNotMatch(result.content, /Intermediate Value Theorem/);
  assert.match(calls[0].sql, /SELECT slug, title, credit, grade_level/);
});

test('a query returns snippets, and full files only when asked', async () => {
  const brief = await searchCurriculum({ query: 'limits' }, { db: db([CALC]) });
  assert.match(brief.content, /matches for "limits"/);
  assert.match(brief.content, /full: true/);

  const whole = await searchCurriculum({ query: 'limits', full: true }, { db: db([CALC]) });
  assert.match(whole.content, /Intermediate Value Theorem/);
  assert.doesNotMatch(whole.content, /full: true/);
});

test('a subject narrows a query instead of being ignored', async () => {
  const calls = [];
  await searchCurriculum({ query: 'limits', subject: 'math' }, { db: db([CALC], calls) });
  assert.match(calls[0].sql, /AND subject = \?3/);
  assert.equal(calls[0].binds.at(-1), 'math');
});

test('missing arguments and a missing database fail clearly', async () => {
  await assert.rejects(() => searchCurriculum({}, { db: db([]) }), /course, a subject, or a query/);
  await assert.rejects(() => searchCurriculum({ course: 'x' }, {}), /not available in this deployment/);
  await assert.rejects(() => searchCurriculum({ course: 'nope' }, { db: db([]) }), /no course matches/);
});

test('wildcards in user input cannot widen the search', async () => {
  const calls = [];
  await searchCurriculum({ query: '%' }, { db: db([CALC], calls) });
  // A bare % would otherwise match every course in the table.
  assert.equal(calls[0].binds[0], '%%');
});

test('executeTool routes the tool and labels its failures', async () => {
  const ok = await executeTool('search_curriculum', { course: 'AP Calculus AB' }, fetch, { db: db([CALC]) });
  assert.equal(ok.error, undefined);
  assert.equal(ok.display.label, 'Curriculum');

  const failed = await executeTool('search_curriculum', {}, fetch, { db: db([]) });
  assert.equal(failed.error, true);
  assert.equal(failed.display.label, 'Curriculum');
});
