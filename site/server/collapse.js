// Detect a reply that has collapsed into cycling the same few dozen words.
//
// A live chat on run5b asked for an AP Calculus unit-circle review and got a
// good start, then a run-on paragraph that drifted into school calendars, then
// hundreds of words cycling "triangle square rectangle parallelogram rhombus
// trapezoid polygon shape figure..." until the token limit. The frequency
// penalty stops a sentence repeating verbatim; it does nothing about a loop
// through a vocabulary. Brittain Code has a guard for this and the web chat
// had none.
//
// THE MEASURE
// The last 200 words of PROSE using fewer than 30% distinct words. Prose is
// the reply with code fences, inline code, math and table rows blanked out:
// those legitimately reuse a small vocabulary -- CSS properties, a derivatives
// table, a Yes/No column -- and must never trip it. Blanking keeps every
// index, so the cut maps straight back onto the reply.
//
// Checked before it was ported, in training/evals: it flags the real collapse
// and none of 21 long legitimate replies from the same model -- a full
// HTML/CSS page, a 25-row Yes/No table, a LaTeX formula sheet, a 900-word
// essay, an 80-item list.
//
// THE CUT
// At the last paragraph break before the collapsed window. The real collapse
// was one paragraph that rambled for hundreds of words before it looped, so
// cutting at the paragraph drops the ramble with the loop. That kept the
// first 3,823 of 23,028 characters, which is everything that was worth
// reading.

const WINDOW = 200;
const FLOOR = 0.3;
const MASKS = [
  /```[\s\S]*?(?:```|$)/g,       // fenced code, closed or still streaming
  /`[^`\n]*`/g,                   // inline code
  /\$\$[\s\S]*?(?:\$\$|$)/g,      // display math
  /\$[^$\n]*\$/g,                 // inline math
  /\\\[[\s\S]*?(?:\\\]|$)/g,      // \[ ... \]
  /\\\([\s\S]*?(?:\\\)|$)/g,      // \( ... \)
  /\\[A-Za-z]+/g,                 // stray LaTeX commands
  /^[ \t]*\|.*$/gm,               // table rows
];

function mask(text) {
  return MASKS.reduce((out, pattern) => out.replace(pattern, match => ' '.repeat(match.length)), text);
}

// Where to cut the reply, or -1 if it has not collapsed.
export function collapsePoint(value) {
  const text = String(value || '');
  const words = [];
  for (const match of mask(text).matchAll(/[A-Za-z']+/g)) {
    words.push([match.index, match[0].toLowerCase()]);
  }
  if (words.length < WINDOW) return -1;
  const tail = words.slice(-WINDOW);
  if (new Set(tail.map(([, word]) => word)).size / WINDOW >= FLOOR) return -1;
  const start = tail[0][0];
  const paragraph = text.lastIndexOf('\n\n', start - 2);
  if (paragraph > 0) return paragraph;
  const sentence = Math.max(...['. ', '! ', '? ', '\n'].map(end => text.lastIndexOf(end, start - end.length)));
  return sentence > 0 ? sentence + 1 : 0;
}

export const COLLAPSE_NOTE = '*(Stopped here: the rest of this reply had started repeating itself.)*';
