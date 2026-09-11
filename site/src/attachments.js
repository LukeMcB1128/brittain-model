export const ACCEPTED_ATTACHMENTS = '.txt,.md,.markdown,.csv,.tsv,.json,.js,.jsx,.ts,.tsx,.py,.java,.c,.cpp,.h,.hpp,.rs,.go,.html,.css,.xml,.yaml,.yml,.toml,.sql,.sh,.log,.pdf,image/png,image/jpeg,image/webp';
export const MAX_ATTACHMENT_COUNT = 5;
export const MAX_ATTACHMENT_BYTES = 10_000_000;

const MAX_IMAGE_BYTES = 5_000_000;
const MAX_TEXT_BYTES = 2_000_000;
const MAX_TEXT_CHARS = 120_000;
const MAX_PDF_PAGES = 50;
const MAX_RENDERED_PDF_PAGES = 4;
const TEXT_EXTENSIONS = new Set(['txt', 'md', 'markdown', 'csv', 'tsv', 'json', 'js', 'jsx', 'ts', 'tsx', 'py', 'java', 'c', 'cpp', 'h', 'hpp', 'rs', 'go', 'html', 'css', 'xml', 'yaml', 'yml', 'toml', 'sql', 'sh', 'log']);
const IMAGE_TYPES = new Set(['image/png', 'image/jpeg', 'image/webp']);

function extension(name) {
  return String(name).split('.').pop()?.toLowerCase() || '';
}

function safeName(name) {
  return [...String(name || 'attachment')].filter(character => {
    const code = character.charCodeAt(0);
    return code >= 32 && code !== 127;
  }).join('').slice(0, 180) || 'attachment';
}

function readDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(`Could not read ${file.name}.`));
    reader.readAsDataURL(file);
  });
}

async function readPdf(file) {
  const [{ getDocument, GlobalWorkerOptions }, worker] = await Promise.all([
    import('pdfjs-dist/build/pdf.mjs'),
    import('pdfjs-dist/build/pdf.worker.min.mjs?url'),
  ]);
  GlobalWorkerOptions.workerSrc = worker.default;
  const bytes = new Uint8Array(await file.arrayBuffer());
  const loadingTask = getDocument({ data: bytes, isEvalSupported: false });
  const document = await loadingTask.promise;
  const sections = [];
  const pageImages = [];
  let length = 0;
  const pageCount = document.numPages;
  const pages = Math.min(document.numPages, MAX_PDF_PAGES);
  try {
    for (let pageNumber = 1; pageNumber <= pages && (pageNumber <= MAX_RENDERED_PDF_PAGES || length < MAX_TEXT_CHARS); pageNumber += 1) {
      const page = await document.getPage(pageNumber);
      const content = await page.getTextContent();
      const text = content.items.map(item => item.str).join(' ').replace(/\s+/g, ' ').trim();
      if (text) {
        const section = `Page ${pageNumber}\n${text}`;
        sections.push(section);
        length += section.length;
      }
      if (pageNumber <= MAX_RENDERED_PDF_PAGES) {
        try {
          const initial = page.getViewport({ scale: 1 });
          const scale = Math.min(1.5, 1200 / Math.max(initial.width, initial.height));
          const viewport = page.getViewport({ scale });
          const canvas = window.document.createElement('canvas');
          canvas.width = Math.ceil(viewport.width);
          canvas.height = Math.ceil(viewport.height);
          const canvasContext = canvas.getContext('2d', { alpha: false });
          await page.render({ canvas, canvasContext, viewport, background: '#ffffff' }).promise;
          pageImages.push({ page: pageNumber, dataUrl: canvas.toDataURL('image/jpeg', 0.82) });
        } catch { /* Text extraction still makes the PDF useful when rendering fails. */ }
      }
    }
  } finally {
    if (typeof loadingTask.destroy === 'function') await loadingTask.destroy();
    else if (typeof document.cleanup === 'function') await document.cleanup();
  }
  if (!sections.length && !pageImages.length) throw new Error(`${file.name} could not be read or rendered.`);
  const content = sections.length ? sections.join('\n\n').slice(0, MAX_TEXT_CHARS) : '[No selectable text. Use the rendered page images.]';
  const rawDataUrl = await readDataUrl(file);
  return { content, dataUrl: rawDataUrl.replace(/^data:[^;]*;/, 'data:application/pdf;'), pageCount, pageImages, truncated: pageCount > pages || length > MAX_TEXT_CHARS };
}

export async function importAttachment(file) {
  const name = safeName(file.name);
  if (IMAGE_TYPES.has(file.type)) {
    if (file.size > MAX_IMAGE_BYTES) throw new Error(`${name} is larger than the 5 MB image limit.`);
    return { id: crypto.randomUUID(), name, type: file.type, size: file.size, kind: 'image', dataUrl: await readDataUrl(file) };
  }
  if (file.type === 'application/pdf' || extension(name) === 'pdf') {
    if (file.size > MAX_ATTACHMENT_BYTES) throw new Error(`${name} is larger than the 10 MB PDF limit.`);
    const extracted = await readPdf(file);
    return { id: crypto.randomUUID(), name, type: 'application/pdf', size: file.size, kind: 'pdf', ...extracted };
  }
  if (file.type.startsWith('text/') || TEXT_EXTENSIONS.has(extension(name))) {
    if (file.size > MAX_TEXT_BYTES) throw new Error(`${name} is larger than the 2 MB text-file limit.`);
    const text = await file.text();
    return { id: crypto.randomUUID(), name, type: file.type || 'text/plain', size: file.size, kind: 'text', content: text.slice(0, MAX_TEXT_CHARS), truncated: text.length > MAX_TEXT_CHARS };
  }
  throw new Error(`${name} is not a supported image, PDF, text, code, CSV, JSON, or Markdown file.`);
}

export function messageContent(prompt, attachments = []) {
  if (!attachments.length) return prompt;
  return [
    { type: 'text', text: prompt.trim() || 'Review the attached content.' },
    ...attachments.flatMap(attachment => attachment.kind === 'image'
      ? [{ type: 'text', text: `Attached image: ${safeName(attachment.name)}` }, { type: 'image_url', image_url: { url: attachment.dataUrl } }]
      : [{ type: 'text', text: `Attached file: ${safeName(attachment.name)}${attachment.truncated ? ' (content truncated)' : ''}\n\n${attachment.content}` }]),
  ];
}

export function requestAssets(turns = []) {
  const assets = [];
  const seen = new Set();
  for (const item of turns.flatMap(turn => [...(turn.attachments || []), ...(turn.artifacts || [])])) {
    const supported = item?.kind === 'pdf' || (item?.kind === 'image' && ['image/png', 'image/jpeg'].includes(item.type));
    if (!item?.id || seen.has(item.id) || !item.dataUrl || !supported) continue;
    seen.add(item.id);
    assets.push({ id: item.id, name: item.name, type: item.type, dataUrl: item.dataUrl, ...(item.kind === 'pdf' ? { pageCount: item.pageCount, pageImages: item.pageImages || [] } : {}) });
  }
  return assets;
}
