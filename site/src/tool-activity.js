export function toolActivityMessage(tools) {
  const activeTool = Array.isArray(tools)
    ? tools.findLast(tool => tool?.status === 'running')
    : null;

  if (!activeTool) return '';
  if (activeTool.name === 'web_search' || activeTool.name === 'web_fetch') return 'Searching the web…';
  if (activeTool.name === 'calculate') return 'Calculating…';
  if (activeTool.name?.startsWith('pdf_')) return 'Reading the PDF…';
  return 'Working…';
}
