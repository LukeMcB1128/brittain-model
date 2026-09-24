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

// remark-math treats $$...$$ inside a paragraph as inline math. Move that
// form, and tall expressions that crowd a line of text, into display math.
// Keep short formulas inline so ordinary prose still reads naturally.
function remarkReadableMath() {
  const flowContainers = new Set(['root', 'blockquote', 'listItem', 'footnoteDefinition']);
  const tallMath = /\\(?:dfrac|cfrac|binom|dbinom|int|iint|sum|prod|lim)\b|\\begin\{(?:aligned|align|matrix|pmatrix|bmatrix|cases|array|split|gathered)\*?\}/;
  return (tree, file) => {
    const source = String(file.value);
    // A new math node needs the same HTML shape as one parsed from a $$ block.
    // Without it, rehype renders the TeX source as plain text.
    const displayNode = (part, value) => ({
      type: 'math',
      value,
      position: part.position,
      data: {
        hName: 'pre',
        hChildren: [{ type: 'element', tagName: 'code', properties: { className: ['language-math', 'math-display'] }, children: [{ type: 'text', value }] }],
      },
    });
    const shouldDisplay = node => {
      if (node.type !== 'inlineMath') return false;
      const offset = node.position?.start?.offset;
      if (offset !== undefined && source.slice(offset, offset + 2) === '$$') return true;
      return tallMath.test(node.value)
        || (node.value.includes('\\frac') && (node.value.length >= 20 || /[+\-=^]/.test(node.value)))
        || node.value.length > 48;
    };
    function visit(container) {
      if (!container?.children) return;
      const children = [];
      for (const child of container.children) {
        if (child.type === 'paragraph' && flowContainers.has(container.type) && child.children?.some(shouldDisplay)) {
          let inline = [];
          const flush = () => {
            if (inline.some(node => node.type !== 'text' || node.value.trim())) {
              children.push({ type: 'paragraph', children: inline });
            }
            inline = [];
          };
          for (let index = 0; index < child.children.length; index += 1) {
            const part = child.children[index];
            if (shouldDisplay(part)) {
              flush();
              const next = child.children[index + 1];
              const punctuation = next?.type === 'text' && /^[.,;:]$/.test(next.value.trim()) ? next.value.trim() : '';
              children.push(displayNode(part, part.value + (punctuation ? `\\text{${punctuation}}` : '')));
              if (punctuation) index += 1;
            } else inline.push(part);
          }
          flush();
        } else {
          visit(child);
          children.push(child);
        }
      }
      container.children = children;
    }
    visit(tree);
  };
}

const components = {
  a: ({ children, href }) => h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, children),
  // Model-supplied image URLs stay links, so viewing a reply does not load external images.
  img: ({ alt, src }) => h('a', { href: src, target: '_blank', rel: 'noopener noreferrer' }, alt || 'View image'),
  table: ({ children }) => h('div', { className: 'c-table-scroll', tabIndex: 0, role: 'region', 'aria-label': 'Response table' }, h('table', null, children)),
};
export default function MarkdownReply({ text }) {
  return h('div', { className: 'c-markdown' }, h(Markdown, {
    remarkPlugins: [remarkGfm, remarkMath, remarkCurrencyDollars, remarkReadableMath],
    rehypePlugins: [[rehypeKatex, { strict: 'ignore', throwOnError: false }]],
    skipHtml: true,
    components,
  }, text));
}
