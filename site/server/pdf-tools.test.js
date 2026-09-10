import test from 'node:test';
import assert from 'node:assert/strict';
import { PDFDocument } from 'pdf-lib';
import { executePdfTool, parsePdfPages, PDF_TOOL_DEFINITIONS } from './pdf-tools.js';

function dataUrl(bytes, type = 'application/pdf') {
  return `data:${type};base64,${Buffer.from(bytes).toString('base64')}`;
}

async function samplePdf(name, pages = 2, withForm = false) {
  const document = await PDFDocument.create();
  for (let index = 0; index < pages; index += 1) document.addPage([612, 792]);
  if (withForm) {
    const field = document.getForm().createTextField('full_name');
    field.addToPage(document.getPage(0), { x: 40, y: 700, width: 200, height: 24 });
  }
  const bytes = await document.save();
  return { id: name, name, type: 'application/pdf', dataUrl: dataUrl(bytes), pageCount: pages, pageImages: [] };
}

test('defines all six attachment-scoped PDF tools and parses page ranges', () => {
  assert.deepEqual(PDF_TOOL_DEFINITIONS.map(tool => tool.function.name), ['pdf_info', 'pdf_render', 'pdf_fill_form', 'pdf_stamp', 'pdf_pages', 'pdf_merge']);
  assert.deepEqual(parsePdfPages('1-2,4,4', 5), [0, 1, 3]);
  assert.throws(() => parsePdfPages('6', 5), /outside/);
});

test('inspects, stamps, extracts, and merges attached PDFs in memory', async () => {
  const first = await samplePdf('first.pdf', 2);
  const second = await samplePdf('second.pdf', 1);
  const context = { pdfs: [first, second], images: [] };
  const info = await executePdfTool('pdf_info', { file: 'first.pdf' }, context);
  assert.equal(JSON.parse(info.content).page_count, 2);
  const stamped = await executePdfTool('pdf_stamp', { file: 'first.pdf', page: 1, x: 20, y: 20, text: 'Approved' }, context);
  assert.equal(stamped.artifact.name, 'first-stamped.pdf');
  const extracted = await executePdfTool('pdf_pages', { file: 'first.pdf', operation: 'extract', pages: '2' }, context);
  assert.match(extracted.content, /Extracted 1 page/);
  const merged = await executePdfTool('pdf_merge', { files: ['first.pdf', 'second.pdf'] }, context);
  const mergedBytes = Buffer.from(merged.artifact.dataUrl.split(',')[1], 'base64');
  assert.equal((await PDFDocument.load(mergedBytes)).getPageCount(), 3);
});

test('fills exact form fields and returns a new downloadable PDF', async () => {
  const formPdf = await samplePdf('form.pdf', 1, true);
  const output = await executePdfTool('pdf_fill_form', { file: 'form.pdf', values: { full_name: 'Luke Brittain' } }, { pdfs: [formPdf], images: [] });
  const bytes = Buffer.from(output.artifact.dataUrl.split(',')[1], 'base64');
  const document = await PDFDocument.load(bytes);
  assert.equal(document.getForm().getTextField('full_name').getText(), 'Luke Brittain');
});

test('returns prepared PDF page images to the model', async () => {
  const pdf = await samplePdf('scan.pdf', 2);
  pdf.pageImages = [{ page: 1, dataUrl: 'data:image/jpeg;base64,YQ==' }];
  const output = await executePdfTool('pdf_render', { file: 'scan.pdf', pages: '1' }, { pdfs: [pdf], images: [] });
  assert.deepEqual(output.modelImages, ['data:image/jpeg;base64,YQ==']);
});
