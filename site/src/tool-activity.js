const ACTIONS = {
  web_search: { icon: 'search', running: 'Searching the web', done: 'Searched the web', error: 'Web search failed' },
  web_fetch: { icon: 'search', running: 'Reading a web page', done: 'Read a web page', error: 'Web page read failed' },
  calculate: { icon: 'calculate', running: 'Calculating', done: 'Calculated', error: 'Calculation failed' },
  search_curriculum: { icon: 'search', running: 'Searching the curriculum', done: 'Checked the curriculum', error: 'Curriculum search failed' },
  pdf_info: { icon: 'file', running: 'Inspecting the PDF', done: 'Inspected the PDF', error: 'PDF inspection failed' },
  pdf_render: { icon: 'file', running: 'Reading PDF pages', done: 'Read PDF pages', error: 'PDF page read failed' },
  pdf_fill_form: { icon: 'file', running: 'Filling the PDF form', done: 'Filled the PDF form', error: 'PDF form fill failed' },
  pdf_stamp: { icon: 'file', running: 'Editing the PDF', done: 'Edited the PDF', error: 'PDF edit failed' },
  pdf_pages: { icon: 'file', running: 'Updating PDF pages', done: 'Updated PDF pages', error: 'PDF page update failed' },
  pdf_merge: { icon: 'file', running: 'Merging PDFs', done: 'Merged PDFs', error: 'PDF merge failed' },
};

function clipped(value, limit) {
  return String(value || '').replace(/[\r\n]+/g, ' ').trim().slice(0, limit);
}

function actionFor(name) {
  if (ACTIONS[name]) return ACTIONS[name];
  if (String(name).startsWith('pdf_')) return { icon: 'file', running: 'Working with the PDF', done: 'Updated the PDF', error: 'PDF action failed' };
  return { icon: 'model', running: 'Using a tool', done: 'Tool completed', error: 'Tool failed' };
}

export function toolActivityItems(tools, compactionStatus = '') {
  const items = [];
  if (['running', 'done', 'error'].includes(compactionStatus)) {
    items.push({
      id: 'compaction',
      icon: 'compact',
      status: compactionStatus,
      label: compactionStatus === 'running' ? 'Compacting earlier context'
        : compactionStatus === 'done' ? 'Earlier context compacted'
          : 'Context compaction failed',
      detail: compactionStatus === 'done' ? 'Older turns were summarized for this reply.'
        : compactionStatus === 'error' ? 'The full conversation was used.' : '',
      result: '',
    });
  }
  if (!Array.isArray(tools)) return items;
  for (const [index, tool] of tools.slice(-15).entries()) {
    if (!tool?.name || !['running', 'done', 'error'].includes(tool.status)) continue;
    const action = actionFor(tool.name);
    items.push({
      id: tool.id || `${tool.name}-${index}`,
      icon: action.icon,
      status: tool.status,
      label: action[tool.status],
      detail: clipped(tool.detail, 160),
      result: tool.status === 'running' ? '' : clipped(tool.result, 120),
    });
  }
  return items;
}

export function activityProgressMessage(status, phase) {
  if (status !== 'streaming') return '';
  if (phase === 'compacting' || phase === 'tool' || phase === 'answering') return '';
  if (phase === 'reviewing') return 'Reviewing results…';
  return 'Thinking…';
}
