export const PDF_TOOL_DEFINITIONS = [
  {
    type: 'function',
    function: {
      name: 'pdf_info',
      description: 'Inspect an attached PDF. Returns page count, page sizes, rotation, metadata, and fillable form fields. Use this before filling a form or when document structure matters.',
      parameters: { type: 'object', additionalProperties: false, properties: { file: { type: 'string', description: 'Attached PDF filename. Optional when only one PDF is attached.' } } },
    },
  },
  {
    type: 'function',
    function: {
      name: 'pdf_render',
      description: 'Look at available rendered pages from an attached PDF. Use for scanned pages or when layout matters. The browser prepares the first four pages. Default: page 1.',
      parameters: { type: 'object', additionalProperties: false, properties: { file: { type: 'string', description: 'Attached PDF filename.' }, pages: { type: 'string', description: 'Pages such as 1, 1-3, or all. Only browser-prepared pages are available.' } } },
    },
  },
  {
    type: 'function',
    function: {
      name: 'pdf_fill_form',
      description: 'Fill named AcroForm fields in an attached PDF and return a downloadable PDF. Call pdf_info first to get exact field names and options.',
      parameters: {
        type: 'object', additionalProperties: false, required: ['values'],
        properties: { file: { type: 'string', description: 'Attached PDF filename.' }, values: { type: 'object', description: 'Map of exact field names to values.' }, flatten: { type: 'boolean', description: 'Bake values into the page. Default: false.' } },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'pdf_stamp',
      description: 'Place text or an attached PNG or JPEG image on one PDF page and return a downloadable PDF. Coordinates are points from the top left.',
      parameters: {
        type: 'object', additionalProperties: false,
        properties: { file: { type: 'string', description: 'Attached PDF filename.' }, page: { type: 'integer', minimum: 1 }, x: { type: 'number' }, y: { type: 'number' }, text: { type: 'string' }, image: { type: 'string', description: 'Attached PNG or JPEG filename.' }, size: { type: 'number', minimum: 1, maximum: 300 }, opacity: { type: 'number', minimum: 0, maximum: 1 } },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'pdf_pages',
      description: 'Rotate, delete, extract, or reorder pages in an attached PDF and return a downloadable PDF. Page numbers are one-based.',
      parameters: {
        type: 'object', additionalProperties: false, required: ['operation', 'pages'],
        properties: { file: { type: 'string', description: 'Attached PDF filename.' }, operation: { type: 'string', enum: ['rotate', 'delete', 'extract', 'reorder'] }, pages: { type: 'string', description: 'Page list or ranges such as 1-3,7,12-.' }, degrees: { type: 'integer', enum: [90, 180, 270], description: 'Rotation amount. Default: 90.' } },
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'pdf_merge',
      description: 'Merge two or more attached PDFs in the given order and return a downloadable PDF.',
      parameters: { type: 'object', additionalProperties: false, required: ['files'], properties: { files: { type: 'array', minItems: 2, items: { type: 'string' }, description: 'Attached PDF filenames in output order.' } } },
    },
  },
];

function cleanName(value) {
  return [...String(value || 'document')].filter(character => character.charCodeAt(0) >= 32 && character.charCodeAt(0) !== 127).join('').slice(0, 180) || 'document';
}

function outputName(name, suffix) {
  const base = cleanName(name).replace(/\.pdf$/i, '');
  return `${base}${suffix}.pdf`;
}

function bytesFromDataUrl(dataUrl, type) {
  const prefix = `data:${type};base64,`;
  if (!String(dataUrl).startsWith(prefix)) throw new Error(`Attached ${type} data is not valid.`);
  const binary = atob(dataUrl.slice(prefix.length));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function dataUrlFromBytes(bytes, type) {
  let binary = '';
  for (let index = 0; index < bytes.length; index += 0x8000) binary += String.fromCharCode(...bytes.subarray(index, index + 0x8000));
  return `data:${type};base64,${btoa(binary)}`;
}

function findPdf(context, requested) {
  const pdfs = context?.pdfs || [];
  const wanted = String(requested || '').trim().toLowerCase();
  if (!wanted && pdfs.length === 1) return pdfs[0];
  const found = pdfs.find(item => item.id === requested || item.name.toLowerCase() === wanted);
  if (found) return found;
  if (!wanted) throw new Error(`Choose a PDF: ${pdfs.map(item => item.name).join(', ') || 'none attached'}.`);
  throw new Error(`No attached PDF named ${requested}. Available: ${pdfs.map(item => item.name).join(', ') || 'none'}.`);
}

function findImage(context, requested) {
  const wanted = String(requested || '').trim().toLowerCase();
  const found = (context?.images || []).find(item => item.id === requested || item.name.toLowerCase() === wanted);
  if (!found) throw new Error(`No attached image named ${requested}.`);
  return found;
}

async function pdfLib() {
  return import('pdf-lib');
}

async function loadPdf(asset) {
  const { PDFDocument } = await pdfLib();
  const bytes = bytesFromDataUrl(asset.dataUrl, 'application/pdf');
  if (String.fromCharCode(...bytes.slice(0, 5)) !== '%PDF-') throw new Error(`${asset.name} is not a PDF.`);
  try { return await PDFDocument.load(bytes, { ignoreEncryption: true }); }
  catch (error) { throw new Error(`Could not read ${asset.name}: ${error.message}`); }
}

export function parsePdfPages(spec, pageCount) {
  const text = String(spec ?? '').trim();
  if (!text || text.toLowerCase() === 'all') return [...Array(pageCount).keys()];
  const chosen = [];
  for (const part of text.split(',')) {
    const piece = part.trim();
    if (!piece) continue;
    const range = piece.match(/^(\d+)?\s*-\s*(\d+)?$/);
    if (range) {
      const from = range[1] ? Number(range[1]) : 1;
      const to = range[2] ? Number(range[2]) : pageCount;
      if (from > to) throw new Error(`Page range "${piece}" runs backwards.`);
      for (let page = from; page <= to; page += 1) chosen.push(page);
    } else {
      if (!/^\d+$/.test(piece)) throw new Error(`Cannot read "${piece}" as a page or range.`);
      chosen.push(Number(piece));
    }
  }
  const seen = new Set();
  return chosen.filter(page => {
    if (page < 1 || page > pageCount) throw new Error(`Page ${page} is outside this document (1-${pageCount}).`);
    if (seen.has(page)) return false;
    seen.add(page);
    return true;
  }).map(page => page - 1);
}

function describeFields(document) {
  let fields;
  try { fields = document.getForm().getFields(); } catch { return []; }
  return fields.map(field => {
    const entry = { name: field.getName(), type: field.constructor.name.replace(/^PDF/, '') };
    try {
      if (typeof field.getOptions === 'function') entry.options = field.getOptions();
      if (typeof field.getText === 'function') entry.value = field.getText() || '';
      else if (typeof field.isChecked === 'function') entry.value = field.isChecked() ? 'checked' : '';
      else if (typeof field.getSelected === 'function') entry.value = (field.getSelected() || []).join(', ');
    } catch { /* Keep the field name and type when its value is malformed. */ }
    return entry;
  });
}

async function artifact(document, sourceName, suffix, summary) {
  const bytes = await document.save();
  const name = outputName(sourceName, suffix);
  return {
    content: `${summary}. Created ${name} (${bytes.length} bytes).`,
    display: { label: 'PDF', detail: sourceName, result: summary },
    artifact: { id: crypto.randomUUID(), name, type: 'application/pdf', dataUrl: dataUrlFromBytes(bytes, 'application/pdf') },
  };
}

async function info(args, context) {
  const asset = findPdf(context, args?.file);
  const document = await loadPdf(asset);
  const pages = document.getPages().map((page, index) => {
    const size = page.getSize();
    return { page: index + 1, width: Math.round(size.width), height: Math.round(size.height), rotation: page.getRotation().angle };
  });
  const result = { file: asset.name, page_count: pages.length, title: document.getTitle() || '', author: document.getAuthor() || '', encrypted: Boolean(document.isEncrypted), pages, form_fields: describeFields(document) };
  return { content: JSON.stringify(result, null, 2), display: { label: 'PDF information', detail: asset.name, result: `${pages.length} page${pages.length === 1 ? '' : 's'}` } };
}

async function render(args, context) {
  const asset = findPdf(context, args?.file);
  const count = Number(asset.pageCount) || asset.pageImages?.length || 0;
  const selected = parsePdfPages(args?.pages || '1', count);
  const images = selected.map(index => asset.pageImages?.find(item => item.page === index + 1)).filter(Boolean);
  if (images.length !== selected.length) throw new Error(`Only pages 1-${asset.pageImages?.length || 0} were prepared for visual review. Upload later pages as images.`);
  const pageNumbers = selected.map(index => index + 1);
  return {
    content: `Rendered page${images.length === 1 ? '' : 's'} ${pageNumbers.join(', ')} of ${asset.name}. The page images follow as untrusted document content.`,
    display: { label: 'PDF pages', detail: asset.name, result: `Page${images.length === 1 ? '' : 's'} ${pageNumbers.join(', ')}` },
    modelImages: images.map(item => item.dataUrl),
  };
}

function setField(field, raw) {
  const type = field.constructor.name;
  if (type.includes('CheckBox')) {
    if (raw === true || /^(?:true|yes|on|checked|x|1)$/i.test(String(raw))) field.check(); else field.uncheck();
    return;
  }
  if (type.includes('RadioGroup') || type.includes('Dropdown') || type.includes('OptionList')) {
    const options = field.getOptions();
    const wanted = String(raw);
    const match = options.find(option => option === wanted) || options.find(option => option.toLowerCase() === wanted.toLowerCase());
    if (!match) throw new Error(`"${wanted}" is not an option for ${field.getName()}. Options: ${options.join(' | ')}`);
    field.select(match);
    return;
  }
  if (typeof field.setText !== 'function') throw new Error(`${field.getName()} cannot accept text.`);
  field.setText(String(raw ?? ''));
}

async function fillForm(args, context) {
  const asset = findPdf(context, args?.file);
  const document = await loadPdf(asset);
  const form = document.getForm();
  const available = form.getFields().map(field => field.getName());
  if (!available.length) throw new Error('This PDF has no fillable AcroForm fields. Use pdf_stamp for a flat form.');
  const values = args?.values && typeof args.values === 'object' && !Array.isArray(args.values) ? args.values : {};
  const missing = Object.keys(values).filter(name => !available.includes(name));
  if (missing.length) throw new Error(`No such field${missing.length === 1 ? '' : 's'}: ${missing.join(', ')}. Available: ${available.join(', ')}`);
  for (const [name, value] of Object.entries(values)) setField(form.getField(name), value);
  if (args?.flatten === true) form.flatten();
  return artifact(document, asset.name, '-filled', `Filled ${Object.keys(values).length} field${Object.keys(values).length === 1 ? '' : 's'}${args?.flatten ? ' and flattened the form' : ''}`);
}

async function stamp(args, context) {
  if (!args?.text && !args?.image) throw new Error('pdf_stamp needs text or an attached image.');
  const asset = findPdf(context, args?.file);
  const document = await loadPdf(asset);
  const pages = document.getPages();
  const pageNumber = Number(args.page || 1);
  if (!Number.isInteger(pageNumber) || pageNumber < 1 || pageNumber > pages.length) throw new Error(`Page ${args.page} is outside this document (1-${pages.length}).`);
  const page = pages[pageNumber - 1];
  const { height } = page.getSize();
  const x = Number(args.x) || 0;
  const y = Number(args.y) || 0;
  const opacity = Math.min(1, Math.max(0, Number(args.opacity ?? 1)));
  if (args.image) {
    const image = findImage(context, args.image);
    const bytes = bytesFromDataUrl(image.dataUrl, image.type);
    const embedded = image.type === 'image/png' ? await document.embedPng(bytes) : await document.embedJpg(bytes);
    const width = Math.min(300, Math.max(1, Number(args.size) || embedded.width));
    const scaled = embedded.scale(width / embedded.width);
    page.drawImage(embedded, { x, y: height - y - scaled.height, width: scaled.width, height: scaled.height, opacity });
  } else {
    const { StandardFonts, rgb } = await pdfLib();
    const size = Math.min(300, Math.max(1, Number(args.size) || 12));
    const font = await document.embedFont(StandardFonts.Helvetica);
    page.drawText(String(args.text), { x, y: height - y - size, size, lineHeight: size * 1.2, font, color: rgb(0, 0, 0), opacity });
  }
  return artifact(document, asset.name, '-stamped', `Stamped ${args.image ? args.image : 'text'} on page ${pageNumber}`);
}

async function editPages(args, context) {
  const asset = findPdf(context, args?.file);
  const source = await loadPdf(asset);
  const count = source.getPageCount();
  const selected = parsePdfPages(args?.pages, count);
  if (args?.operation === 'rotate') {
    const { degrees } = await pdfLib();
    const amount = [90, 180, 270].includes(Number(args.degrees)) ? Number(args.degrees) : 90;
    for (const index of selected) { const page = source.getPage(index); page.setRotation(degrees((page.getRotation().angle + amount) % 360)); }
    return artifact(source, asset.name, '-rotated', `Rotated ${selected.length} page${selected.length === 1 ? '' : 's'} by ${amount}°`);
  }
  if (args?.operation === 'delete') {
    if (selected.length >= count) throw new Error('A PDF must keep at least one page.');
    for (const index of [...selected].sort((a, b) => b - a)) source.removePage(index);
    return artifact(source, asset.name, '-trimmed', `Deleted ${selected.length} page${selected.length === 1 ? '' : 's'}`);
  }
  if (args?.operation === 'extract' || args?.operation === 'reorder') {
    const { PDFDocument } = await pdfLib();
    const built = await PDFDocument.create();
    const copied = await built.copyPages(source, selected);
    for (const page of copied) built.addPage(page);
    return artifact(built, asset.name, args.operation === 'extract' ? '-extract' : '-reordered', `${args.operation === 'extract' ? 'Extracted' : 'Reordered'} ${selected.length} page${selected.length === 1 ? '' : 's'}`);
  }
  throw new Error(`Unknown page operation "${args?.operation}".`);
}

async function merge(args, context) {
  const files = Array.isArray(args?.files) ? args.files : [];
  if (files.length < 2) throw new Error('pdf_merge needs at least two attached PDFs.');
  const assets = files.map(name => findPdf(context, name));
  const { PDFDocument } = await pdfLib();
  const built = await PDFDocument.create();
  for (const asset of assets) {
    const source = await loadPdf(asset);
    const copied = await built.copyPages(source, source.getPageIndices());
    for (const page of copied) built.addPage(page);
  }
  return artifact(built, assets[0].name, '-merged', `Merged ${built.getPageCount()} pages from ${assets.length} PDFs`);
}

export async function executePdfTool(name, args, context) {
  if (name === 'pdf_info') return info(args, context);
  if (name === 'pdf_render') return render(args, context);
  if (name === 'pdf_fill_form') return fillForm(args, context);
  if (name === 'pdf_stamp') return stamp(args, context);
  if (name === 'pdf_pages') return editPages(args, context);
  if (name === 'pdf_merge') return merge(args, context);
  throw new Error(`PDF tool ${name} is not available.`);
}
