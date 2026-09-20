import { createElement as h } from 'react';
import Markdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';

// Single dollar signs are useful for inline LaTeX, but a list of prices can
// look like one long equation to remark-math. Restore spans that contain plain
// prose to text after Markdown parsing. Real formulas and LaTeX commands stay
// as math nodes.
function remarkCurrencyDollars() {
  const priceFollowedByProse = /^\s*\d[\d,]*(?:\.\d+)?(?:[,.;:]|\s)+(?:.*[A-Za-z]{2,})/;
  function visit(node) {
    if (!node?.children) return;
    for (const child of node.children) {
      if (child.type === 'inlineMath' && !child.value.includes('\\') && priceFollowedByProse.test(child.value)) {
        child.type = 'text';
        child.value = `$${child.value}$`;
        delete child.data;
      } else visit(child);
    }
  }
  return visit;
}

const components = {
  a: ({ children, href }) => h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, children),
  // Model-supplied image URLs stay links, so viewing a reply does not load external images.
  img: ({ alt, src }) => h('a', { href: src, target: '_blank', rel: 'noopener noreferrer' }, alt || 'View image'),
  table: ({ children }) => h('div', { className: 'c-table-scroll', tabIndex: 0, role: 'region', 'aria-label': 'Response table' }, h('table', null, children)),
};
export default function MarkdownReply({ text }) {
  return h('div', { className: 'c-markdown' }, h(Markdown, {
    remarkPlugins: [remarkGfm, remarkMath, remarkCurrencyDollars],
    rehypePlugins: [[rehypeKatex, { strict: 'ignore', throwOnError: false }]],
    skipHtml: true,
    components,
  }, text));
}
