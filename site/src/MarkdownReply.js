import { createElement as h } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const components = {
  a: ({ children, href }) => h('a', { href, target: '_blank', rel: 'noopener noreferrer' }, children),
  // Model-supplied image URLs stay links, so viewing a reply does not load external images.
  img: ({ alt, src }) => h('a', { href: src, target: '_blank', rel: 'noopener noreferrer' }, alt || 'View image'),
  table: ({ children }) => h('div', { className: 'c-table-scroll', tabIndex: 0, role: 'region', 'aria-label': 'Response table' }, h('table', null, children)),
};
export default function MarkdownReply({ text }) {
  return h('div', { className: 'c-markdown' }, h(Markdown, { remarkPlugins: [remarkGfm], skipHtml: true, components }, text));
}
