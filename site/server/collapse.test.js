import test from 'node:test';
import assert from 'node:assert/strict';
import { collapsePoint } from './collapse.js';

// Distinct pseudo-words, so a fixture of varied prose really is varied: the
// measure only counts letters, and "word1 word2" would all read as "word".
function varied(count) {
  const letters = n => String.fromCharCode(97 + (n % 26)) + String.fromCharCode(97 + (Math.floor(n / 26) % 26));
  return Array.from({ length: count }, (_, n) => `${letters(n)}${letters(n + 7)}`).join(' ');
}
const LOOP = Array(80).fill('triangle square rectangle parallelogram rhombus').join(' ');

test('a reply ending in a vocabulary loop is cut at the paragraph before it', () => {
  // The shape of the real collapse: a good paragraph, then a run-on paragraph
  // that rambles and then cycles a few words to the token limit.
  const good = 'The unit circle has radius one and is centred at the origin.';
  const reply = `${good}\n\nThis part requires careful algebraic work, and then ${LOOP}`;
  // Cutting at the paragraph drops the ramble along with the loop.
  assert.equal(collapsePoint(reply), good.length);
});

test('without a paragraph break it cuts at the last sentence end', () => {
  const reply = `First sentence here. Second one too. ${LOOP}`;
  assert.equal(reply.slice(0, collapsePoint(reply)), 'First sentence here. Second one too.');
});

test('varied prose is left alone, however long', () => {
  assert.equal(collapsePoint(varied(600)), -1);
});

test('text too short to judge is left alone', () => {
  assert.equal(collapsePoint(Array(30).fill('again').join(' ')), -1);
});

test('code, math and tables never trip it', () => {
  // Each of these reuses a small vocabulary on purpose. Checked against real
  // long replies from the served model before this was written: a full
  // HTML/CSS page, a 25-row Yes/No table and a LaTeX formula sheet, none
  // flagged.
  const css = '```css\n' + Array(200).fill('.card { color: red; margin: 0; padding: 0; }').join('\n') + '\n```';
  const streaming = '```css\n' + Array(200).fill('.card { color: red; margin: 0; }').join('\n');
  const table = '| Language | Typed | Compiled |\n|---|---|---|\n'
    + Array(60).fill('| Python | Yes | No |').join('\n');
  const math = Array(120).fill('$\\sin x$ and $\\cos x$').join(' ');
  const latex = Array(120).fill('\\(\\frac{d}{dx}\\sin x = \\cos x\\)').join(' ');
  for (const [name, text] of Object.entries({ css, streaming, table, math, latex })) {
    assert.equal(collapsePoint(`Here it is.\n\n${text}`), -1, name);
  }
});

test('a loop that begins after code is still caught', () => {
  const code = '```js\nconst answer = 42;\n```';
  const reply = `Here is the code.\n\n${code}\n\nNow the explanation ${LOOP}`;
  const cut = collapsePoint(reply);
  assert.ok(cut > 0);
  assert.match(reply.slice(0, cut), /const answer = 42/);
  assert.doesNotMatch(reply.slice(0, cut), /rhombus/);
});
