import test from 'node:test';
import assert from 'node:assert/strict';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import MarkdownReply from './MarkdownReply.js';
import { contextUsage } from './context-usage.js';
const render = text => renderToStaticMarkup(createElement(MarkdownReply, { text }));
test('replies render headings, lists, tables, fenced code and inline styles', () => {
  const html = render('# Heading\n\n**Bold** and *italic* with `code`.\n\n- First\n- Second\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```js\nconst x = 1;\n```');
  for (const tag of ['<h1>', '<strong>', '<em>', '<ul>', '<li>', '<table>', '<pre>', 'language-js']) assert.ok(html.includes(tag), tag);
});
test('model HTML and unsafe links cannot run and remote images are not loaded', () => {
  const html = render('<script>alert(1)</script>\n\n[bad](javascript:alert%281%29)\n\n![remote](https://example.com/image.png)');
  assert.ok(!html.includes('<script'));
  assert.ok(!html.includes('javascript:'));
  assert.ok(!html.includes('<img'));
  assert.ok(html.includes('rel="noopener noreferrer"'));
});
test('partial fenced code can render during streaming', () => {
  assert.ok(render('```python\nprint("Hello').includes('<pre>'));
});
test('inline and display LaTeX render as formatted math', () => {
  const html = render('For $y = -x^2 + 3$, the derivative is:\n\n$$\n\\frac{dy}{dx} = -2x\n$$');
  const sameLine = render('Derivative: $$ \\frac{dy}{dx} = -2x $$');
  assert.match(html, /class="katex"/);
  assert.match(html, /class="katex-display"/);
  assert.match(html, /<mfrac>/);
  assert.doesNotMatch(html, /\$y = -x\^2 \+ 3\$/);
  assert.match(sameLine, /<mfrac>/);
  assert.doesNotMatch(sameLine, /\$\$/);
});
test('context uses latest server total without double-counting history', () => {
  assert.equal(contextUsage([{usage:{total_tokens:100}}, {usage:{total_tokens:250}}]).tokens,250);
  assert.equal(contextUsage([]).tokens,0);
  assert.equal(contextUsage([{status:'streaming'}]).tokens,null);
  const stopped = contextUsage([{usage:{total_tokens:100}}, {status:'stopped'}]);
  assert.equal(stopped.tokens,100);
  assert.equal(stopped.stale,true);
  assert.equal(contextUsage([{usage:{total_tokens:16384}}]).percent,50);
  assert.equal(contextUsage([{usage:{total_tokens:80}}],100).percent,80);
});
