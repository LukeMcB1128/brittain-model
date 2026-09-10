import { useEffect, useRef, useState } from 'react';
import './chat-page.css';
import BrandLogo from './BrandLogo.jsx';
import { readCompletion } from './brittain4.js';
import MarkdownReply from './MarkdownReply.js';
import { contextUsage } from './context-usage.js';
import { ACCEPTED_ATTACHMENTS, importAttachment, MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_COUNT, messageContent } from './attachments.js';

const suggestions = [
  ['write', 'Write something', 'Help me write a clear introduction for a community science project.'],
  ['code', 'Work on code', 'Write a Python function that removes duplicates while keeping their order.'],
  ['explain', 'Understand a topic', 'Explain how a language model uses context. Start with a simple example.'],
  ['plan', 'Make a plan', 'Help me plan a small website. Ask about the audience and the main task first.'],
  ['search', 'Search the web', 'Search the web for the latest news about open-source language models and summarize the main developments with source links.'],
];
function Icon({ name }) {
  const paths = {
    panel: 'M4 4h16v16H4zM9 4v16',
    edit: 'M12 5H5v14h14v-7M14 4l6 6M10 14l1-5 7-7 4 4-7 7-5 1Z',
    arrow: 'M12 19V5m-6 6 6-6 6 6',
    down: 'm7 10 5 5 5-5',
    model: 'M4 5h16v14H4zM8 9h8M8 13h5',
    write: 'm5 19 3-7L18 2l4 4-10 10-7 3ZM8 12l4 4',
    code: 'm8 7-5 5 5 5m8-10 5 5-5 5M14 5l-4 14',
    explain: 'M9 18h6M10 21h4M8 14a6 6 0 1 1 8 0l-1 2H9l-1-2Z',
    plan: 'M8 6h12M8 12h12M8 18h12M3 6h1M3 12h1M3 18h1',
    search: 'm21 21-4.4-4.4M19 11a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
    calculate: 'M5 3h14v18H5zM8 7h8M8 11h2M14 11h2M8 15h2M14 15h2',
    delete: 'M4 7h16M9 7V4h6v3m3 0-1 14H7L6 7m4 4v6m4-6v6',
    attach: 'M21.4 11.6 12 21a6 6 0 0 1-8.5-8.5l10-10a4 4 0 0 1 5.7 5.7l-10 10a2 2 0 1 1-2.9-2.8l9.5-9.5',
    file: 'M6 2h8l4 4v16H6zM14 2v5h5',
    close: 'm7 7 10 10M17 7 7 17',
  };
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.model}/></svg>;
}
function AttachmentCards({ attachments, onRemove }) {
  if (!attachments?.length) return null;
  return <div className={`c-attachments ${onRemove ? 'c-attachments-editable' : ''}`}>{attachments.map(attachment => <div className="c-attachment" key={attachment.id}>{attachment.kind === 'image' ? <img src={attachment.dataUrl} alt=""/> : <span className="c-attachment-file"><Icon name="file"/></span>}<span><strong>{attachment.name}</strong><small>{attachment.kind === 'image' ? 'Image' : attachment.type === 'application/pdf' ? 'PDF text' : 'File text'}{attachment.truncated ? ' · Truncated' : ''}</small></span>{onRemove && <button type="button" onClick={() => onRemove(attachment.id)} aria-label={`Remove ${attachment.name}`}><Icon name="close"/></button>}</div>)}</div>;
}
function ToolActivity({ tools }) {
  if (!tools?.length) return null;
  return <div className="c-tool-list" aria-label="Tools used">{tools.map(tool => {
    const label = tool.label || (tool.name === 'web_search' ? 'Web search' : tool.name === 'web_fetch' ? 'Web page' : 'Calculator');
    const detail = tool.detail || tool.result || (tool.status === 'running' ? 'Working…' : '');
    const status = tool.status === 'running' ? 'Working' : tool.status === 'error' ? 'Failed' : tool.result || 'Done';
    return <div className={`c-tool c-tool-${tool.status}`} key={tool.id}><span className="c-tool-icon"><Icon name={tool.name === 'calculate' ? 'calculate' : 'search'}/></span><span><strong>{label}</strong><small>{detail}</small></span><span className="c-tool-status">{status}</span></div>;
  })}</div>;
}
export default function Chat() {
  const [sidebar, setSidebar] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [info, setInfo] = useState(false);
  const [draft, setDraft] = useState('');
  const [chats, setChats] = useState([]);
  const [active, setActive] = useState(null);
  const [connection, setConnection] = useState('loading');
  const [busy, setBusy] = useState(false);
  const [contextLimit, setContextLimit] = useState(32768);
  const [copyNotice, setCopyNotice] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [attachmentError, setAttachmentError] = useState('');
  const [importing, setImporting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const controller = useRef(null);
  const input = useRef(null);
  const fileInput = useRef(null);
  async function checkConnection() {
    setConnection('loading');
    try {
      const response = await fetch('/api/session', { signal: AbortSignal.timeout(10000) });
      if (response.status === 401) { setConnection('signed-out'); return; }
      if (!response.ok) throw new Error();
      const data = await response.json();
      if (Number.isInteger(data.context) && data.context > 0) setContextLimit(data.context);
      setConnection(data.ready ? 'ready' : data.configured ? 'offline' : 'unconfigured');
    } catch { setConnection('offline'); }
  }
  useEffect(() => { checkConnection(); return () => controller.current?.abort(); }, []);
  const preview = connection === 'ready';
  const end = useRef(null);
  const current = chats.find(chat => chat.id === active);
  const messages = current?.messages || [];
  const context = contextUsage(messages, contextLimit);
  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }); }, [chats, active]);
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') { setSidebar(false); setInfo(false); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  function clearImports() { setAttachments([]); setAttachmentError(''); setDragging(false); }
  function newChat() { if (controller.current) return; setActive(null); setDraft(''); clearImports(); setSidebar(false); input.current?.focus(); }
  function deleteChat(id) {
    if (controller.current) return;
    setChats(items => items.filter(chat => chat.id !== id));
    setCopyNotice('Conversation deleted.');
    if (active === id) { setActive(null); setDraft(''); clearImports(); setSidebar(false); requestAnimationFrame(() => input.current?.focus()); }
  }
  async function addFiles(fileList) {
    if (busy || importing || !fileList?.length) return;
    const selected = [...fileList];
    const room = MAX_ATTACHMENT_COUNT - attachments.length;
    if (room <= 0) { setAttachmentError(`You can attach up to ${MAX_ATTACHMENT_COUNT} files.`); return; }
    setImporting(true);
    setAttachmentError('');
    const added = [];
    const errors = [];
    let total = attachments.reduce((sum, item) => sum + item.size, 0);
    for (const file of selected.slice(0, room)) {
      if (total + file.size > MAX_ATTACHMENT_BYTES) { errors.push('Attachments must total 10 MB or less.'); break; }
      try { const item = await importAttachment(file); added.push(item); total += file.size; }
      catch (error) { errors.push(error.message); }
    }
    if (selected.length > room) errors.push(`Only ${MAX_ATTACHMENT_COUNT} attachments are allowed.`);
    setAttachments(items => [...items, ...added]);
    setAttachmentError([...new Set(errors)].join(' '));
    setImporting(false);
    if (fileInput.current) fileInput.current.value = '';
  }
  async function send(e, retry = false) {
    e?.preventDefault();
    if (controller.current || importing || connection !== 'ready' || (!retry && !draft.trim() && !attachments.length)) return;
    const id = active || crypto.randomUUID();
    const turnAttachments = retry ? messages.at(-1).attachments || [] : attachments;
    const prompt = retry ? messages.at(-1).prompt : draft.trim() || 'Review the attached content.';
    const turn = { id: crypto.randomUUID(), prompt, attachments: turnAttachments, answer: '', tools: [], status: 'streaming' };
    const previous = retry ? messages.slice(0, -1) : messages;
    const history = previous.flatMap(m => [{ role: 'user', content: messageContent(m.prompt, m.attachments) }, ...(m.answer ? [{ role: 'assistant', content: m.answer }] : [])]);
    if (active) setChats(items => items.map(chat => chat.id === id ? { ...chat, messages: [...previous, turn] } : chat));
    else { setChats(items => [{ id, title: (draft.trim() || turnAttachments[0]?.name || turn.prompt).slice(0, 60), messages: [turn] }, ...items]); setActive(id); }
    if (!retry) { setDraft(''); clearImports(); }
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    function update(patch) { setChats(items => items.map(chat => chat.id === id ? { ...chat, messages: chat.messages.map(m => m.id === turn.id ? { ...m, ...patch } : m) } : chat)); }
    function updateTool(tool) {
      setChats(items => items.map(chat => chat.id === id ? { ...chat, messages: chat.messages.map(message => {
        if (message.id !== turn.id) return message;
        const usedTools = message.tools || [];
        const index = usedTools.findIndex(item => item.id === tool.id);
        return { ...message, tools: index === -1 ? [...usedTools, tool] : usedTools.map((item, toolIndex) => toolIndex === index ? { ...item, ...tool } : item) };
      }) } : chat));
    }
    let answer = '';
    let finishReason;
    try {
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: [...history, { role: 'user', content: messageContent(turn.prompt, turnAttachments) }] }), signal: abort.signal });
      await readCompletion(response, chunk => {
        if (chunk.type === 'tool') { updateTool(chunk); return; }
        if (chunk.type === 'usage') { update({ usage: chunk.usage }); return; }
        answer += chunk.text || '';
        if (chunk.finishReason) finishReason = chunk.finishReason;
        update({ answer, ...(chunk.usage ? { usage: chunk.usage } : {}) });
      });
      update({ status: 'done', note: finishReason === 'length' ? 'The reply reached its output limit. Ask the model to continue.' : !answer ? 'The model returned no text. Please retry.' : '' });
    } catch (error) {
      update({ status: abort.signal.aborted ? 'stopped' : 'error', error: abort.signal.aborted ? '' : error.message });
    } finally { controller.current = null; setBusy(false); input.current?.focus(); }
  }
  async function copyAnswer(answer) {
    try { await navigator.clipboard.writeText(answer); setCopyNotice('Reply copied.'); }
    catch { setCopyNotice('Could not copy. Select the reply and copy it manually.'); }
  }
  const composer = <div className="c-composer-wrap">
    <form className={`c-composer ${dragging ? 'c-dragging' : ''}`} onSubmit={send} onDragEnter={e => { e.preventDefault(); setDragging(true); }} onDragOver={e => e.preventDefault()} onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget)) setDragging(false); }} onDrop={e => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}>
      <input ref={fileInput} hidden type="file" multiple accept={ACCEPTED_ATTACHMENTS} onChange={e => addFiles(e.target.files)}/>
      <AttachmentCards attachments={attachments} onRemove={id => setAttachments(items => items.filter(item => item.id !== id))}/>
      <label className="sr-only" htmlFor="chat-message">Message Brittain 4</label>
      <textarea ref={input} id="chat-message" placeholder="Message Brittain 4…" value={draft} onChange={e => setDraft(e.target.value)} rows={2} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) send(e); }}/>
      <div className="c-composer-controls"><button type="button" className="c-attach" disabled={busy || importing || attachments.length >= MAX_ATTACHMENT_COUNT} onClick={() => fileInput.current?.click()} aria-label={importing ? 'Adding attachments' : 'Add files or images'} title="Add files or images"><Icon name="attach"/></button><span>{importing ? 'Adding…' : attachments.length ? `${attachments.length} / ${MAX_ATTACHMENT_COUNT}` : 'Files and images'}</span>{busy ? <button type="button" className="c-send" onClick={() => controller.current?.abort()} aria-label="Stop reply">■</button> : <button type="submit" className="c-send" disabled={(!draft.trim() && !attachments.length) || !preview || importing} aria-label="Send message"><Icon name="arrow"/></button>}</div>
      {dragging && <div className="c-drop-message">Drop files to attach</div>}
    </form>
    {attachmentError && <p className="c-attachment-error" role="alert">{attachmentError}</p>}
    <div className="c-context" title="Server-reported prompt and reply tokens from the latest request. Excludes your unsent message. Counts are not added across requests.">
      <div className="c-context-label"><span>{context.stale && context.tokens !== null ? 'Last reported context' : 'Context used'}</span><span>{context.tokens === null ? 'Awaiting usage' : `${context.tokens.toLocaleString()} / ${context.limit.toLocaleString()} tokens (${context.percent}%)`}</span></div>
      <progress max={context.limit} value={context.tokens === null ? 0 : Math.min(context.limit, context.tokens)} aria-label="Context used" />
      {(busy || context.stale || draft.trim()) && <p>{busy ? '' : context.stale ? 'Latest reply usage is unavailable.' : ''}</p>}
    </div>
    <p className="c-disclaimer">{connection === 'loading' ? 'Checking chat connection…' : preview ? 'Chats stay in this page session. Check important information.' : connection === 'signed-out' ? 'Sign in to this private site to use chat.' : connection === 'unconfigured' ? 'The server access key is not configured yet.' : 'Cannot reach the chat service.'}{!preview && connection !== 'loading' && <button type="button" className="c-inline-button" onClick={checkConnection}>Retry connection</button>}</p>
  </div>;
  return <main className={`c-app ${collapsed ? 'c-collapsed' : ''} ${sidebar ? 'c-sidebar-open' : ''}`}>
    {sidebar && <button className="c-scrim" onClick={() => setSidebar(false)} aria-label="Close navigation"/>}
    <aside className="c-sidebar" aria-label="Chat navigation">
      <div className="c-sidebar-top"><a href="#/" className="c-brand" aria-label="Brittain home"><BrandLogo/>BRITTAIN</a><button className="c-icon c-desktop" aria-label="Collapse sidebar" onClick={() => setCollapsed(true)}><Icon name="panel"/></button><button className="c-icon c-mobile" aria-label="Close sidebar" onClick={() => setSidebar(false)}><Icon name="panel"/></button></div>
      <button className="c-nav-item" disabled={busy} onClick={newChat}><Icon name="edit"/>New chat</button>
      <a href="#/models" className="c-nav-item"><Icon name="model"/>Models</a>
      <div className="c-history"><h2>Conversations</h2>{chats.length ? chats.map(chat => <div key={chat.id} className={`c-history-row ${active === chat.id ? 'c-active' : ''}`}><button className="c-history-item" disabled={busy} aria-current={active === chat.id ? 'page' : undefined} onClick={() => { setActive(chat.id); setSidebar(false); setDraft(''); clearImports(); }}>{chat.title}</button><button className="c-history-delete" disabled={busy} onClick={() => deleteChat(chat.id)} aria-label={`Delete conversation: ${chat.title}`} title="Delete conversation"><Icon name="delete"/></button></div>) : <p>Your chats will appear here.</p>}</div>
      <div className="c-sidebar-bottom">{preview ? <div className="c-account"><span className="c-avatar">B</span><div>Private chat<small>Local to this page</small></div></div> : <><p>Try Brittain 4 with a free account.</p><a className="button full-width" href="#/signup">Sign up free</a></>}<a href="#/" className="c-home-link">← Back to Brittain</a></div>
    </aside>
    <section className="c-main">
      <header className="c-header"><div className="c-header-left"><button className={`c-icon c-open ${collapsed ? 'c-is-collapsed' : ''}`} aria-label="Open sidebar" aria-expanded={sidebar || !collapsed} onClick={() => { setCollapsed(false); setSidebar(true); }}><Icon name="panel"/></button><div className="c-model-wrap"><button className="c-model-button" onClick={() => setInfo(!info)} aria-expanded={info}>Brittain 4<Icon name="down"/></button>{info && <div className="c-model-info"><strong>Brittain 4</strong><p>9B dense model</p><dl><div><dt>Web chat</dt><dd>{contextLimit.toLocaleString()} tokens</dd></div><div><dt>Model maximum</dt><dd>262k</dd></div></dl><a href="#/models/brittain-4">View model details ↗</a></div>}</div></div><div className="c-header-right"><span className="c-preview">{busy ? 'Responding…' : preview ? 'Connected' : 'Not connected'}</span>{!preview && <a href="#/signup" className="button compact">Sign up free</a>}</div></header>
      {messages.length === 0 ? <div className="c-start"><div className="c-start-inner"><h1>What can I help with?</h1>{composer}<div className="c-suggestions">{suggestions.map(([icon,label,prompt]) => <button key={icon} onClick={() => { setDraft(prompt); input.current?.focus(); }}><Icon name={icon}/>{label}</button>)}</div></div></div> : <><div className="c-conversation" role="log" aria-label="Conversation"><div className="c-message-column">{messages.map((message, index) => <div className="c-turn" key={message.id}><div className="c-user-message"><AttachmentCards attachments={message.attachments}/><span>{message.prompt}</span></div><div className="c-reply"><BrandLogo/><div className="c-response"><ToolActivity tools={message.tools}/>{message.answer ? <MarkdownReply text={message.answer}/> : message.status === 'streaming' ? <p role="status">{message.tools?.some(tool => tool.status === 'running') ? 'Using tools…' : 'Waiting for Brittain 4…'}</p> : null}{message.error && <p className="c-error" role="alert">{message.error}</p>}{message.status === 'stopped' && <p>Reply stopped.</p>}{message.note && <p>{message.note}</p>}<div className="c-reply-actions">{message.answer && <button onClick={() => copyAnswer(message.answer)}>Copy</button>}{!busy && index === messages.length - 1 && <button onClick={e => send(e, true)}>Retry</button>}</div></div></div></div>)}<span className="sr-only" role="status">{copyNotice}</span><div ref={end}/></div></div><div className="c-bottom-composer">{composer}</div></>}
    </section>
  </main>;
}
