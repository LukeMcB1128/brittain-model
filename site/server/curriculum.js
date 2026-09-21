// Look up AISD/TEA course curriculum from D1.
//
// WHY THIS IS A TOOL AND NOT TRAINING DATA
// TEKS codes, PEIMS numbers and course numbers must be exact and are revised
// yearly. A model that has memorised them offers no way to tell a remembered
// code from an invented one, and a wrong TEKS citation reads exactly like a
// right one. So the curriculum is queried and quoted, never recalled.
//
// WHY RESULTS ARE NOT LABELLED UNTRUSTED
// web_search and web_fetch return content from the internet, and the system
// prompt tells the model to treat that as evidence and ignore instructions
// inside it. This is the opposite: it is the district's own curriculum, loaded
// from a repository, and its wording is the point. The framing says so, and
// asks for verbatim quotation, because the failure mode here is a paraphrased
// student expectation -- wrong in a way that reads as right.
//
// WHY WHOLE DOCUMENTS
// A course averages 5.8 KB, about 1,450 tokens, so the whole file fits in
// context. No chunking and no embeddings: the retrieval unit is the document a
// teacher would open.

// A full course is ~5.8 KB. Three of them plus a reply is comfortable inside a
// 32k window; more crowds out the conversation that asked for them.
const MAX_FULL = 3;
const MAX_LIST = 40;
const MAX_MATCHES = 5;
const SNIPPET_CHARS = 400;
const MAX_CONTENT_CHARS = 26_000;

const HEADER = 'AISD/TEA curriculum, from the district course files. This is '
  + 'authoritative source text, not web content: quote TEKS language and course '
  + 'codes verbatim rather than paraphrasing them, and cite the course file.';

export const CURRICULUM_TOOL = {
  type: 'function',
  function: {
    name: 'search_curriculum',
    description: 'Look up Austin ISD high school course curriculum aligned to Texas '
      + 'TEKS: course descriptions, units of study, credit, grade level, course '
      + 'numbers, PEIMS codes and TEKS citations. Use for any question about what '
      + 'a course covers or how courses are sequenced. Call with no arguments to '
      + 'see every subject and how many courses each has.',
    parameters: {
      type: 'object', additionalProperties: false,
      properties: {
        course: { type: 'string', description: 'A course name or slug, e.g. "AP Calculus AB" or "math/ap-calculus-ab". Returns that course in full.' },
        subject: { type: 'string', description: 'One of: math, science, english, social-studies, world-languages, fine-arts, health-pe, cte, innovative. Lists the courses in it.' },
        query: { type: 'string', description: 'Words to search for across all courses.' },
        full: { type: 'boolean', description: 'Return complete course files rather than snippets. Default false.' },
      },
    },
  },
};

function clip(text, limit) {
  return text.length > limit ? `${text.slice(0, limit)}\n[...truncated]` : text;
}

function courseBlock(row) {
  return [
    `## ${row.title}  (${row.slug})`,
    `Credit: ${row.credit || 'n/a'} | Grade level: ${row.grade_level || 'n/a'}`,
    `Course number: ${row.course_number || 'n/a'} | PEIMS: ${row.peims || 'n/a'}`,
    `TEKS: ${row.teks_cite || 'n/a'}`,
    '',
    row.body,
  ].join('\n');
}

// The word nearest the match, so a list of hits shows why each one matched.
function snippet(body, needle) {
  const at = body.toLowerCase().indexOf(needle.toLowerCase());
  if (at < 0) return body.slice(0, SNIPPET_CHARS).trim();
  const from = Math.max(0, at - SNIPPET_CHARS / 3);
  return `${from ? '...' : ''}${body.slice(from, from + SNIPPET_CHARS).trim()}...`;
}

export async function searchCurriculum(args, context = {}, options = {}) {
  const db = context.db;
  if (!db || typeof db.prepare !== 'function') {
    throw new Error('the curriculum database is not available in this deployment');
  }
  const course = String(args?.course || '').trim();
  const subject = String(args?.subject || '').trim().toLowerCase();
  const query = String(args?.query || '').trim();
  const full = args?.full === true;
  // No arguments is a legitimate question -- "what is in here?" -- and the
  // database answers it completely. Before this, a vague query returned
  // "nothing matches" and the model went to the open web for a catalogue it
  // already had.
  if (!course && !subject && !query) {
    const { results } = await db.prepare(
      'SELECT subject, COUNT(*) AS n FROM courses GROUP BY subject ORDER BY n DESC',
    ).all();
    if (!results?.length) throw new Error('the curriculum database is empty');
    const total = results.reduce((sum, row) => sum + row.n, 0);
    const lines = results.map(row => `- ${row.subject}: ${row.n} courses`);
    return {
      content: `${HEADER}\n\n### ${total} courses across ${results.length} subjects\n`
        + `${lines.join('\n')}\n\nCall again with a subject to list its courses, `
        + `or a course name for the full file.`,
      display: { label: 'Curriculum', detail: 'all subjects', result: `${total} courses` },
    };
  }
  for (const [name, value] of [['course', course], ['subject', subject], ['query', query]]) {
    if (value.length > 200) throw new Error(`${name} is too long`);
  }

  // A named course is the precise case: slug first, then title, then a loose
  // title match, so "AP Calc AB" and "math/ap-calculus-ab" both land.
  if (course) {
    const like = `%${course.replace(/[%_]/g, '')}%`;
    const { results } = await db.prepare(
      'SELECT * FROM courses WHERE slug = ?1 OR lower(title) = lower(?2) '
      + 'OR slug LIKE lower(?3) OR title LIKE ?3 '
      + 'ORDER BY CASE WHEN slug = ?1 OR lower(title) = lower(?2) THEN 0 ELSE 1 END, length(title) '
      + 'LIMIT ?4',
    ).bind(course, course, like, options.limit || MAX_FULL).all();
    if (!results?.length) throw new Error(`no course matches ${course}`);
    const body = `${HEADER}\n\n${results.map(courseBlock).join('\n\n---\n\n')}`;
    return {
      content: clip(body, MAX_CONTENT_CHARS),
      display: { label: 'Curriculum', detail: results[0].title, result: results.length === 1 ? 'Course read' : `${results.length} courses` },
    };
  }

  // A subject alone is a browse: titles only. Returning 178 CTE course files
  // would fill the window and answer nothing.
  if (subject && !query) {
    // The true total is counted separately from the page that is returned.
    // Reporting the capped row count as the total told the model CTE has 40
    // courses when it has 178, and the model repeated that to the user.
    const counted = await db.prepare(
      'SELECT COUNT(*) AS n FROM courses WHERE subject = ?1',
    ).bind(subject).first();
    const total = counted?.n ?? 0;
    if (!total) throw new Error(`no subject named ${subject}`);
    const { results } = await db.prepare(
      'SELECT slug, title, credit, grade_level FROM courses WHERE subject = ?1 ORDER BY title LIMIT ?2',
    ).bind(subject, MAX_LIST).all();
    const lines = results.map(row => `- ${row.title} (${row.slug}) — ${row.credit || 'n/a'}, grades ${row.grade_level || 'n/a'}`);
    const more = total > results.length
      ? `\n\n[${results.length} of ${total} shown. Narrow with a query to see the rest.]`
      : '';
    return {
      content: `${HEADER}\n\n### ${subject} — ${total} courses\n${lines.join('\n')}${more}`,
      display: { label: 'Curriculum', detail: subject, result: `${total} courses` },
    };
  }

  const like = `%${query.replace(/[%_]/g, '')}%`;
  const scoped = subject ? 'AND subject = ?3 ' : '';
  const statement = db.prepare(
    'SELECT * FROM courses WHERE (title LIKE ?1 OR body LIKE ?1) '
    + scoped
    + 'ORDER BY CASE WHEN title LIKE ?1 THEN 0 ELSE 1 END, title LIMIT ?2',
  );
  const { results } = await (subject
    ? statement.bind(like, full ? MAX_FULL : MAX_MATCHES, subject)
    : statement.bind(like, full ? MAX_FULL : MAX_MATCHES)).all();
  if (!results?.length) {
    throw new Error(`nothing in the curriculum matches ${query}`);
  }
  const body = full
    ? results.map(courseBlock).join('\n\n---\n\n')
    : results.map(row => `- **${row.title}** (${row.slug})\n  ${snippet(row.body, query)}`).join('\n\n');
  const note = full ? '' : '\n\nCall again with full: true and a course name for the complete file.';
  return {
    content: clip(`${HEADER}\n\n### matches for "${query}"\n\n${body}${note}`, MAX_CONTENT_CHARS),
    display: { label: 'Curriculum', detail: query, result: `${results.length} match${results.length === 1 ? '' : 'es'}` },
  };
}
