import { useEffect, useRef, useState } from 'react';
import './chat-page.css';
import BrandLogo from './BrandLogo.jsx';
import { readCompletion } from './brittain4.js';
import MarkdownReply from './MarkdownReply.js';
import { contextUsage } from './context-usage.js';
import { withoutReferenceNotes } from './chat.js';
import { ACCEPTED_ATTACHMENTS, chatForSave, importAttachment, MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_COUNT, messageContent, planRequestAttachments } from './attachments.js';
import { assistantToolHistory } from './chat-history.js';
import { activityProgressMessage, toolActivityItems } from './tool-activity.js';

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
    rename: 'M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z',
    check: 'm5 12 4 4L19 6',
    attach: 'M21.4 11.6 12 21a6 6 0 0 1-8.5-8.5l10-10a4 4 0 0 1 5.7 5.7l-10 10a2 2 0 1 1-2.9-2.8l9.5-9.5',
    file: 'M6 2h8l4 4v16H6zM14 2v5h5',
    close: 'm7 7 10 10M17 7 7 17',
    compact: 'M4 9h5V4m11 5h-5V4M4 15h5v5m11-5h-5v5',
  };
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.model}/></svg>;
}
function AttachmentCards({ attachments, onRemove }) {
  if (!attachments?.length) return null;
  return <div className={`c-attachments ${onRemove ? 'c-attachments-editable' : ''}`}>{attachments.map(attachment => <div className="c-attachment" key={attachment.id}>{attachment.kind === 'image' && attachment.dataUrl ? <img src={attachment.dataUrl} alt=""/> : <span className="c-attachment-file"><Icon name="file"/></span>}<span><strong>{attachment.name}</strong><small title={attachment.truncated ? 'Only the first 44,000 characters are added to the model context.' : undefined}>{attachment.kind === 'image' ? 'Image' : attachment.type === 'application/pdf' ? 'PDF' : 'File text'}{attachment.truncated ? ' · Shortened for context' : ''}</small></span>{onRemove && <button type="button" onClick={() => onRemove(attachment.id)} aria-label={`Remove ${attachment.name}`}><Icon name="close"/></button>}</div>)}</div>;
}
function DownloadCards({ artifacts }) {
  if (!artifacts?.length) return null;
  return <div className="c-downloads" aria-label="Generated files">{artifacts.map(file => <a key={file.id} href={file.dataUrl} download={file.name}><span className="c-attachment-file"><Icon name="file"/></span><span><strong>{file.name}</strong><small>Download PDF</small></span></a>)}</div>;
}
// While a reply is in flight this says what is happening right now, one line.
// Afterwards it folds into a single summary the reader can open.
//
// It used to stack a card per tool above every answer, permanently. A reply
// that searched nine subjects left nine cards sitting on top of it, which
// buried the thing the reader actually asked for. The information is worth
// keeping -- which sources, which course files -- but not at the top of the
// page forever.
function ToolActivity({ message }) {
  const items = toolActivityItems(message.tools, message.compactionStatus);
  const progress = activityProgressMessage(message.status, message.activityPhase);
  if (!items.length && !progress) return null;
  const running = items.findLast(item => item.status === 'running');
  const liveMessage = running?.label || progress;

  if (message.status === 'streaming') {
    return <p className="c-tool-live" aria-label="Response activity">
      <span className="c-tool-spinner" aria-hidden="true"/>
      <span className="c-tool-live-copy">
        <strong>{running ? `${running.label}…` : progress}</strong>
        {running?.detail && <small title={running.detail}>{running.detail}</small>}
      </span>
      <span className="sr-only" role="status">{liveMessage}</span>
    </p>;
  }

  const failed = items.filter(item => item.status === 'error').length;
  return <details className="c-tool-activities" aria-label="Response activity">
    <summary className="c-tool-summary">
      <Icon name="search"/>
      <span>{items.length === 1 ? '1 step' : `${items.length} steps`}{failed ? ` · ${failed} failed` : ''}</span>
    </summary>
    {items.map(item => <div className={`c-tool-activity c-tool-${item.status}`} key={item.id}>
      <span className="c-tool-icon"><Icon name={item.icon}/></span>
      <span className="c-tool-copy"><strong>{item.label}</strong>{item.detail && <small title={item.detail}>{item.detail}</small>}</span>
      {item.result ? <span className="c-tool-result" title={item.result}>{item.result}</span> : <Icon name={item.status === 'done' ? 'check' : 'close'}/>}
    </div>)}
  </details>;
}
function groupLabel(dateValue) {
  const date = new Date(dateValue);
  if (Number.isNaN(date.getTime())) return 'Older';
  const today = new Date();
  const start = new Date(today.getFullYear(), today.getMonth(), today.getDate());
  const age = start.getTime() - new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
  if (age <= 0) return 'Today';
  if (age < 7 * 86400000) return 'Previous 7 days';
  return 'Older';
}
export default function Chat({ session, initialChatId = '' }) {
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
  const [historyStatus, setHistoryStatus] = useState('loading');
  const [historyQuery, setHistoryQuery] = useState('');
  const [loadingChat, setLoadingChat] = useState('');
  const [renaming, setRenaming] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [notice, setNotice] = useState('');
  const controller = useRef(null);
  const input = useRef(null);
  const fileInput = useRef(null);
  const deleteTimer = useRef(null);
  const pendingDeleteRef = useRef(null);
  const noticeTimer = useRef(null);
  const modelWrap = useRef(null);
  const sidebarToggle = useRef(null);
  const chatsRef = useRef([]);
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
  async function loadChats() {
    setHistoryStatus('loading');
    try {
      const response = await fetch('/api/chats');
      if (!response.ok) throw new Error();
      const data = await response.json();
      setChats(Array.isArray(data.chats) ? data.chats : []);
      setHistoryStatus('ready');
      if (initialChatId) await loadChat(initialChatId, false, data.chats || []);
    } catch { setHistoryStatus('error'); }
  }
  function setChatUrl(id, replace = false) {
    const url = id ? `/chat/${encodeURIComponent(id)}` : '/chat';
    window.history[replace ? 'replaceState' : 'pushState'](null, '', url);
  }
  async function loadChat(id, updateUrl = true, knownChats = chats) {
    if (controller.current || !id) return;
    const loaded = knownChats.find(chat => chat.id === id && Array.isArray(chat.messages));
    if (loaded) {
      setActive(id); setSidebar(false); setDraft(''); clearImports();
      if (updateUrl) setChatUrl(id);
      return;
    }
    setLoadingChat(id);
    try {
      const response = await fetch(`/api/chats/${encodeURIComponent(id)}`);
      if (!response.ok) throw new Error(response.status === 404 ? 'Conversation not found.' : 'The conversation could not be loaded.');
      const data = await response.json();
      setChats(items => {
        const exists = items.some(chat => chat.id === id);
        return exists ? items.map(chat => chat.id === id ? data.chat : chat) : [data.chat, ...items];
      });
      setActive(id); setSidebar(false); setDraft(''); clearImports();
      if (updateUrl) setChatUrl(id);
    } catch (error) { showNotice(error.message); if (!updateUrl) setChatUrl('', true); }
    finally { setLoadingChat(''); }
  }
  async function saveChat(chat) {
    const response = await fetch(`/api/chats/${encodeURIComponent(chat.id)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(chatForSave(chat)),
    });
    if (!response.ok) {
      let data; try { data = await response.json(); } catch { data = {}; }
      throw new Error(data.error || 'The conversation could not be saved.');
    }
  }
  useEffect(() => { chatsRef.current = chats; }, [chats]);
  // This effect owns the page lifetime and must run once.
  /* oxlint-disable react-hooks/exhaustive-deps */
  useEffect(() => {
    checkConnection(); loadChats();
    const onPopState = () => {
      const match = window.location.pathname.match(/^\/chat\/([^/]+)$/);
      if (match) loadChat(decodeURIComponent(match[1]), false, chatsRef.current);
      else { setActive(null); setDraft(''); clearImports(); }
    };
    window.addEventListener('popstate', onPopState);
    return () => {
      controller.current?.abort();
      window.removeEventListener('popstate', onPopState);
      clearTimeout(deleteTimer.current); clearTimeout(noticeTimer.current);
      if (pendingDeleteRef.current) commitDelete(pendingDeleteRef.current, true);
    };
  }, []);
  /* oxlint-enable react-hooks/exhaustive-deps */
  const preview = connection === 'ready';
  const end = useRef(null);
  const current = chats.find(chat => chat.id === active);
  const messages = current?.messages || [];
  const context = contextUsage(messages, contextLimit);
  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }); }, [chats, active]);
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') { if (sidebar) closeSidebar(); setInfo(false); } };
    const onPointer = e => { if (info && modelWrap.current && !modelWrap.current.contains(e.target)) setInfo(false); };
    window.addEventListener('keydown', onKey);
    window.addEventListener('pointerdown', onPointer);
    return () => { window.removeEventListener('keydown', onKey); window.removeEventListener('pointerdown', onPointer); };
  }, [info, sidebar]);
  function clearImports() { setAttachments([]); setAttachmentError(''); setDragging(false); }
  function showNotice(message, duration = 3500) {
    clearTimeout(noticeTimer.current); setNotice(message);
    noticeTimer.current = setTimeout(() => setNotice(''), duration);
  }
  function closeSidebar() {
    setSidebar(false);
    requestAnimationFrame(() => sidebarToggle.current?.focus());
  }
  function newChat(updateUrl = true) { if (controller.current) return; setActive(null); setDraft(''); clearImports(); setSidebar(false); if (updateUrl) setChatUrl(''); input.current?.focus(); }
  async function commitDelete(item, keepalive = false) {
    try {
      const response = await fetch(`/api/chats/${encodeURIComponent(item.chat.id)}`, { method: 'DELETE', keepalive });
      if (!response.ok) throw new Error();
    } catch {
      if (!keepalive) {
        setChats(items => items.some(chat => chat.id === item.chat.id) ? items : [...items.slice(0, item.index), item.chat, ...items.slice(item.index)]);
        showNotice('The conversation could not be deleted.');
      }
    }
  }
  function deleteChat(id) {
    if (controller.current) return;
    if (pendingDeleteRef.current) { clearTimeout(deleteTimer.current); commitDelete(pendingDeleteRef.current); }
    const index = chats.findIndex(chat => chat.id === id);
    if (index < 0) return;
    const item = { chat: chats[index], index, wasActive: active === id };
    pendingDeleteRef.current = item; setPendingDelete(item);
    setChats(items => items.filter(chat => chat.id !== id));
    if (active === id) newChat();
    deleteTimer.current = setTimeout(() => {
      pendingDeleteRef.current = null; setPendingDelete(null); commitDelete(item); showNotice('Conversation deleted.');
    }, 5000);
  }
  function undoDelete() {
    const item = pendingDeleteRef.current;
    if (!item) return;
    clearTimeout(deleteTimer.current); pendingDeleteRef.current = null; setPendingDelete(null);
    setChats(items => items.some(chat => chat.id === item.chat.id) ? items : [...items.slice(0, item.index), item.chat, ...items.slice(item.index)]);
    if (item.wasActive) { setActive(item.chat.id); setChatUrl(item.chat.id); }
  }
  async function renameChat(e) {
    e.preventDefault();
    const title = renaming?.title.trim();
    if (!title) return;
    try {
      const response = await fetch(`/api/chats/${encodeURIComponent(renaming.id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) });
      if (!response.ok) throw new Error();
      const data = await response.json();
      setChats(items => items.map(chat => chat.id === renaming.id ? { ...chat, ...data.chat } : chat));
      setRenaming(null); showNotice('Conversation renamed.');
    } catch { showNotice('The conversation could not be renamed.'); }
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
    const turn = { id: crypto.randomUUID(), prompt, attachments: turnAttachments, answer: '', tools: [], artifacts: [], status: 'streaming', activityPhase: 'thinking' };
    const previous = retry ? messages.slice(0, -1) : messages;
    const contextStart = Math.min(current?.contextStart || 0, previous.length);
    const contextTurns = previous.slice(contextStart);
    const requestAttachments = planRequestAttachments([...contextTurns, turn]);
    const history = contextTurns.flatMap(m => {
      const tools = assistantToolHistory(m.tools);
      return [{ role: 'user', content: messageContent(m.prompt, m.attachments, requestAttachments.retainedIds), turnId: m.id }, ...(m.answer ? [{ role: 'assistant', content: m.answer, turnId: m.id, ...(tools.length ? { tools } : {}) }] : [])];
    });
    const changedAt = new Date().toISOString();
    let workingChat = current ? { ...current, updatedAt: changedAt, messages: [...previous, turn] } : { id, title: (draft.trim() || turnAttachments[0]?.name || turn.prompt).slice(0, 60), createdAt: changedAt, updatedAt: changedAt, messages: [turn] };
    if (active) setChats(items => items.map(chat => chat.id === id ? workingChat : chat));
    else { setChats(items => [workingChat, ...items]); setActive(id); setChatUrl(id, true); }
    if (!retry) { setDraft(''); clearImports(); }
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    function update(patch) {
      workingChat = { ...workingChat, messages: workingChat.messages.map(m => m.id === turn.id ? { ...m, ...patch } : m) };
      setChats(items => items.map(chat => chat.id === id ? workingChat : chat));
    }
    function updateTool(tool) {
      workingChat = { ...workingChat, messages: workingChat.messages.map(message => {
        if (message.id !== turn.id) return message;
        const usedTools = message.tools || [];
        const index = usedTools.findIndex(item => item.id === tool.id);
        return { ...message, activityPhase: tool.status === 'running' ? 'tool' : 'reviewing', tools: index === -1 ? [...usedTools, tool] : usedTools.map((item, toolIndex) => toolIndex === index ? { ...item, ...tool, detail: tool.detail || item.detail } : item) };
      }) };
      setChats(items => items.map(chat => chat.id === id ? workingChat : chat));
    }
    function updateArtifact(artifact) {
      workingChat = { ...workingChat, messages: workingChat.messages.map(message => message.id === turn.id ? { ...message, artifacts: [...(message.artifacts || []), { ...artifact, type: artifact.mediaType, kind: 'pdf' }] } : message) };
      setChats(items => items.map(chat => chat.id === id ? workingChat : chat));
    }
    let answer = '';
    let finishReason;
    try {
      const savedBeforeReply = {
        ...workingChat,
        messages: workingChat.messages.map(message => message.id === turn.id ? { ...message, status: 'stopped', note: 'Reply interrupted before completion.' } : message),
      };
      try { await saveChat(savedBeforeReply); }
      catch (saveError) { setHistoryStatus('error'); showNotice(saveError.message); }
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: [...history, { role: 'user', content: messageContent(turn.prompt, turnAttachments, requestAttachments.retainedIds), turnId: turn.id }], attachments: requestAttachments.assets, memory: current?.memory || '' }), signal: abort.signal });
      await readCompletion(response, chunk => {
        if (chunk.type === 'tool') { updateTool(chunk); return; }
        if (chunk.type === 'artifact') { updateArtifact(chunk); return; }
        if (chunk.type === 'compaction') {
          const boundary = chunk.throughTurnId ? workingChat.messages.findIndex(message => message.id === chunk.throughTurnId) + 1 : 0;
          workingChat = {
            ...workingChat,
            ...(chunk.status === 'done' ? { memory: chunk.memory, contextStart: Math.max(workingChat.contextStart || 0, boundary) } : {}),
            messages: workingChat.messages.map(message => message.id === turn.id ? { ...message, compactionStatus: chunk.status, activityPhase: chunk.status === 'running' ? 'compacting' : 'thinking' } : message),
          };
          setChats(items => items.map(item => {
            if (item.id !== id) return item;
            return workingChat;
          }));
          return;
        }
        if (chunk.type === 'usage') { update({ usage: chunk.usage }); return; }
        // The server superseded what it has sent: that text was a preamble
        // before a tool call, not part of the answer.
        if (chunk.type === 'reset') { answer = ''; update({ answer }); return; }
        answer += chunk.text || '';
        if (chunk.finishReason) finishReason = chunk.finishReason;
        update({ answer, ...(chunk.text ? { activityPhase: 'answering' } : {}), ...(chunk.usage ? { usage: chunk.usage } : {}) });
      });
      update({ status: 'done', activityPhase: 'done', note: finishReason === 'length' ? 'The reply reached its output limit. Ask the model to continue.' : !answer ? 'The model returned no text. Please retry.' : '' });
    } catch (error) {
      update({ status: abort.signal.aborted ? 'stopped' : 'error', activityPhase: 'done', error: abort.signal.aborted ? '' : error.message });
    } finally {
      try { await saveChat(workingChat); setHistoryStatus('ready'); }
      catch (saveError) { setHistoryStatus('error'); setCopyNotice(saveError.message); }
      controller.current = null; setBusy(false); input.current?.focus();
    }
  }
  async function copyAnswer(answer) {
    try { await navigator.clipboard.writeText(answer); setCopyNotice('Reply copied.'); showNotice('Reply copied.'); }
    catch { setCopyNotice('Could not copy. Select the reply and copy it manually.'); showNotice('Could not copy. Select the reply and copy it manually.'); }
  }
  const visibleChats = chats.filter(chat => chat.title?.toLowerCase().includes(historyQuery.trim().toLowerCase()));
  const chatGroups = visibleChats.reduce((groups, chat) => {
    const label = groupLabel(chat.updatedAt || chat.createdAt);
    (groups[label] ||= []).push(chat);
    return groups;
  }, {});
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
    <p className="c-disclaimer">{connection === 'loading' ? 'Checking chat connection…' : preview ? historyStatus === 'error' ? 'Connected. Chat history could not be saved.' : 'Chats are saved to your account. Check important information.' : connection === 'signed-out' ? 'Sign in to use chat.' : connection === 'unconfigured' ? 'The server access key is not configured yet.' : 'Cannot reach the chat service.'}{!preview && connection !== 'loading' && <button type="button" className="c-inline-button" onClick={checkConnection}>Retry connection</button>}</p>
  </div>;
  return <main className={`c-app ${collapsed ? 'c-collapsed' : ''} ${sidebar ? 'c-sidebar-open' : ''}`}>
    {sidebar && <button className="c-scrim" onClick={closeSidebar} aria-label="Close navigation"/>}
    <aside id="chat-navigation" className="c-sidebar" aria-label="Chat navigation">
      <div className="c-sidebar-top"><a href="/" className="c-brand" aria-label="Brittain home"><BrandLogo/>BRITTAIN</a><button className="c-icon c-desktop" aria-label="Collapse sidebar" onClick={() => { setCollapsed(true); setSidebar(false); requestAnimationFrame(() => sidebarToggle.current?.focus()); }}><Icon name="panel"/></button><button className="c-icon c-mobile" aria-label="Close sidebar" onClick={closeSidebar}><Icon name="panel"/></button></div>
      <button className="c-nav-item" disabled={busy} onClick={newChat}><Icon name="edit"/>New chat</button>
      <a href="/models" className="c-nav-item"><Icon name="model"/>Models</a>
      <div className="c-history"><h2>Conversations</h2><label className="sr-only" htmlFor="chat-search">Search conversations</label><input id="chat-search" className="c-history-search" type="search" placeholder="Search conversations" value={historyQuery} onChange={event => setHistoryQuery(event.target.value)}/>{historyStatus === 'loading' ? <p>Loading chats…</p> : visibleChats.length ? ['Today', 'Previous 7 days', 'Older'].map(label => chatGroups[label]?.length ? <section className="c-history-group" key={label}><h3>{label}</h3>{chatGroups[label].map(chat => <div key={chat.id} className={`c-history-row ${active === chat.id ? 'c-active' : ''}`}>{renaming?.id === chat.id ? <form className="c-rename-form" onSubmit={renameChat}><label className="sr-only" htmlFor={`rename-${chat.id}`}>Conversation name</label><input id={`rename-${chat.id}`} autoFocus maxLength={80} value={renaming.title} onChange={event => setRenaming({ ...renaming, title: event.target.value })} onKeyDown={event => { if (event.key === 'Escape') setRenaming(null); }}/><button type="submit" aria-label="Save name"><Icon name="check"/></button><button type="button" onClick={() => setRenaming(null)} aria-label="Cancel rename"><Icon name="close"/></button></form> : <><button className="c-history-item" disabled={busy || loadingChat === chat.id} aria-current={active === chat.id ? 'page' : undefined} onClick={() => loadChat(chat.id)}>{loadingChat === chat.id ? 'Loading…' : chat.title}</button><button className="c-history-action" disabled={busy} onClick={() => setRenaming({ id: chat.id, title: chat.title })} aria-label={`Rename conversation: ${chat.title}`} title="Rename conversation"><Icon name="rename"/></button><button className="c-history-action c-history-delete" disabled={busy} onClick={() => deleteChat(chat.id)} aria-label={`Delete conversation: ${chat.title}`} title="Delete conversation"><Icon name="delete"/></button></>}</div>)}</section> : null) : <p>{historyQuery ? 'No matching conversations.' : 'Your chats will appear here.'}</p>}</div>
      <div className="c-sidebar-bottom"><a href="/account" className="c-account"><span className="c-avatar">{session.user.name?.slice(0, 1).toUpperCase() || 'B'}</span><div>{session.user.name}<small>{session.user.email}</small></div></a><a href="/" className="c-home-link">← Back to Brittain</a></div>
    </aside>
    <section className="c-main">
      <header className="c-header"><div className="c-header-left"><button ref={sidebarToggle} className={`c-icon c-open ${collapsed ? 'c-is-collapsed' : ''}`} aria-label="Open sidebar" aria-expanded={sidebar} aria-controls="chat-navigation" onClick={() => { setCollapsed(false); setSidebar(true); }}><Icon name="panel"/></button><div className="c-model-wrap" ref={modelWrap}><button className="c-model-button" onClick={() => setInfo(!info)} aria-expanded={info} aria-controls="model-information">Brittain 4<Icon name="down"/></button>{info && <div id="model-information" className="c-model-info" role="region" aria-label="Brittain 4 information"><strong>Brittain 4</strong><p>9B dense model</p><dl><div><dt>Web chat</dt><dd>{contextLimit.toLocaleString()} tokens</dd></div><div><dt>Model maximum</dt><dd>262k</dd></div></dl><a href="/models/brittain-4">View model details ↗</a></div>}</div></div><div className="c-header-right"><span className="c-preview">{busy ? 'Responding…' : preview ? 'Connected' : 'Not connected'}</span></div></header>
      {messages.length === 0 ? <div className="c-start"><div className="c-start-inner"><h1>What can I help with?</h1>{composer}<div className="c-suggestions">{suggestions.map(([icon,label,prompt]) => <button key={icon} onClick={() => { setDraft(prompt); input.current?.focus(); }}><Icon name={icon}/>{label}</button>)}</div></div></div> : <><div className="c-conversation" role="log" aria-label="Conversation"><div className="c-message-column">{messages.map((message, index) => <div className="c-turn" key={message.id}><div className="c-user-message"><AttachmentCards attachments={message.attachments}/><span>{message.prompt}</span></div><div className="c-reply"><BrandLogo/><div className="c-response"><DownloadCards artifacts={message.artifacts}/><ToolActivity message={message}/>{message.answer && <MarkdownReply text={withoutReferenceNotes(message.answer)}/>}{message.error && <p className="c-error" role="alert">{message.error}</p>}{message.status === 'stopped' && <p>Reply stopped.</p>}{message.note && <p>{message.note}</p>}<div className="c-reply-actions">{message.answer && <button onClick={() => copyAnswer(withoutReferenceNotes(message.answer))}>Copy</button>}{!busy && index === messages.length - 1 && <button onClick={e => send(e, true)}>Retry</button>}</div></div></div></div>)}<span className="sr-only" role="status">{copyNotice}</span><div ref={end}/></div></div><div className="c-bottom-composer">{composer}</div></>}
    </section>
    {(notice || pendingDelete) && <div className="c-toast" role="status"><span>{pendingDelete ? `“${pendingDelete.chat.title}” removed.` : notice}</span>{pendingDelete && <button type="button" onClick={undoDelete}>Undo</button>}</div>}
  </main>;
}
