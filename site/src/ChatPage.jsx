import { useEffect, useRef, useState } from 'react';
import './chat-page.css';
import BrandLogo from './BrandLogo.jsx';
import { readCompletion } from './brittain4.js';
import { MessageContent } from './App.jsx';

const suggestions = [
  ['write', 'Write something', 'Help me write a clear introduction for a community science project.'],
  ['code', 'Work on code', 'Write a Python function that removes duplicates while keeping their order.'],
  ['explain', 'Understand a topic', 'Explain how a language model uses context. Start with a simple example.'],
  ['plan', 'Make a plan', 'Help me plan a small website. Ask about the audience and the main task first.'],
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
  };
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.model}/></svg>;
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
  const [copyNotice, setCopyNotice] = useState('');
  const controller = useRef(null);
  const input = useRef(null);
  async function checkConnection() {
    setConnection('loading');
    try {
      const response = await fetch('/api/session', { signal: AbortSignal.timeout(10000) });
      if (response.status === 401) { setConnection('signed-out'); return; }
      if (!response.ok) throw new Error();
      const data = await response.json();
      setConnection(data.ready ? 'ready' : data.configured ? 'offline' : 'unconfigured');
    } catch { setConnection('offline'); }
  }
  useEffect(() => { checkConnection(); return () => controller.current?.abort(); }, []);
  const preview = connection === 'ready';
  const end = useRef(null);
  const current = chats.find(chat => chat.id === active);
  const messages = current?.messages || [];
  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }); }, [chats, active]);
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') { setSidebar(false); setInfo(false); } };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);
  function newChat() { if (controller.current) return; setActive(null); setDraft(''); setSidebar(false); input.current?.focus(); }
  async function send(e, retry = false) {
    e?.preventDefault();
    if (controller.current || connection !== 'ready' || (!retry && !draft.trim())) return;
    const id = active || crypto.randomUUID();
    const turn = { id: crypto.randomUUID(), prompt: retry ? messages.at(-1).prompt : draft.trim(), answer: '', status: 'streaming' };
    const previous = retry ? messages.slice(0, -1) : messages;
    const history = previous.flatMap(m => [{ role: 'user', content: m.prompt }, ...(m.answer ? [{ role: 'assistant', content: m.answer }] : [])]);
    if (active) setChats(items => items.map(chat => chat.id === id ? { ...chat, messages: [...previous, turn] } : chat));
    else { setChats(items => [{ id, title: turn.prompt.slice(0, 60), messages: [turn] }, ...items]); setActive(id); }
    if (!retry) setDraft('');
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    function update(patch) { setChats(items => items.map(chat => chat.id === id ? { ...chat, messages: chat.messages.map(m => m.id === turn.id ? { ...m, ...patch } : m) } : chat)); }
    let answer = '';
    let finishReason;
    try {
      const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ messages: [...history, { role: 'user', content: turn.prompt }] }), signal: abort.signal });
      await readCompletion(response, chunk => {
        answer += chunk.text;
        if (chunk.finishReason) finishReason = chunk.finishReason;
        update({ answer, ...(chunk.usage ? { usage: chunk.usage } : {}) });
      });
      update({ status: 'done', note: finishReason === 'length' ? 'The reply reached its output limit. Ask the model to continue.' : finishReason === 'tool_calls' ? 'This reply requested a tool. Tools are not enabled yet.' : !answer ? 'The model returned no text. Please retry.' : '' });
    } catch (error) {
      update({ status: abort.signal.aborted ? 'stopped' : 'error', error: abort.signal.aborted ? '' : error.message });
    } finally { controller.current = null; setBusy(false); input.current?.focus(); }
  }
  async function copyAnswer(answer) {
    try { await navigator.clipboard.writeText(answer); setCopyNotice('Reply copied.'); }
    catch { setCopyNotice('Could not copy. Select the reply and copy it manually.'); }
  }
  const composer = <div className="c-composer-wrap">
    <form className="c-composer" onSubmit={send}>
      <label className="sr-only" htmlFor="chat-message">Message Brittain 4</label>
      <textarea ref={input} id="chat-message" placeholder="Message Brittain 4…" value={draft} onChange={e => setDraft(e.target.value)} rows={2} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) send(e); }}/>
      <div className="c-composer-controls"><span>Brittain 4 <span className="c-dot">·</span> 32k context</span>{busy ? <button type="button" className="c-send" onClick={() => controller.current?.abort()} aria-label="Stop reply">■</button> : <button type="submit" className="c-send" disabled={!draft.trim() || !preview} aria-label="Send message"><Icon name="arrow"/></button>}</div>
    </form>
    <p className="c-disclaimer">{connection === 'loading' ? 'Checking chat connection…' : preview ? 'Chats stay in this page session. Check important information.' : connection === 'signed-out' ? 'Sign in to this private site to use chat.' : connection === 'unconfigured' ? 'The server access key is not configured yet.' : 'Cannot reach the chat service.'}{!preview && connection !== 'loading' && <button type="button" className="c-inline-button" onClick={checkConnection}>Retry connection</button>}</p>
  </div>;
  return <main className={`c-app ${collapsed ? 'c-collapsed' : ''} ${sidebar ? 'c-sidebar-open' : ''}`}>
    {sidebar && <button className="c-scrim" onClick={() => setSidebar(false)} aria-label="Close navigation"/>}
    <aside className="c-sidebar" aria-label="Chat navigation">
      <div className="c-sidebar-top"><a href="#/" className="c-brand" aria-label="Brittain home"><BrandLogo/>BRITTAIN</a><button className="c-icon c-desktop" aria-label="Collapse sidebar" onClick={() => setCollapsed(true)}><Icon name="panel"/></button><button className="c-icon c-mobile" aria-label="Close sidebar" onClick={() => setSidebar(false)}><Icon name="panel"/></button></div>
      <button className="c-nav-item" disabled={busy} onClick={newChat}><Icon name="edit"/>New chat</button>
      <a href="#/models" className="c-nav-item"><Icon name="model"/>Models</a>
      <div className="c-history"><h2>Conversations</h2>{chats.length ? chats.map(chat => <button key={chat.id} className="c-history-item" disabled={busy} aria-current={active === chat.id ? 'page' : undefined} onClick={() => { setActive(chat.id); setSidebar(false); setDraft(''); }}>{chat.title}</button>) : <p>Your chats will appear here.</p>}</div>
      <div className="c-sidebar-bottom">{preview ? <div className="c-account"><span className="c-avatar">B</span><div>Private chat<small>Local to this page</small></div></div> : <><p>Try Brittain 4 with a free account.</p><a className="button full-width" href="#/signup">Sign up free</a></>}<a href="#/" className="c-home-link">← Back to Brittain</a></div>
    </aside>
    <section className="c-main">
      <header className="c-header"><div className="c-header-left"><button className={`c-icon c-open ${collapsed ? 'c-is-collapsed' : ''}`} aria-label="Open sidebar" aria-expanded={sidebar || !collapsed} onClick={() => { setCollapsed(false); setSidebar(true); }}><Icon name="panel"/></button><div className="c-model-wrap"><button className="c-model-button" onClick={() => setInfo(!info)} aria-expanded={info}>Brittain 4<Icon name="down"/></button>{info && <div className="c-model-info"><strong>Brittain 4</strong><p>9B dense model</p><dl><div><dt>Web chat</dt><dd>32,768 tokens</dd></div><div><dt>Model maximum</dt><dd>262k</dd></div></dl><a href="#/models/brittain-4">View model details ↗</a></div>}</div></div><div className="c-header-right"><span className="c-preview">{busy ? 'Responding…' : preview ? 'Connected' : 'Not connected'}</span>{!preview && <a href="#/signup" className="button compact">Sign up free</a>}</div></header>
      {messages.length === 0 ? <div className="c-start"><div className="c-start-inner"><h1>What can I help with?</h1>{composer}<div className="c-suggestions">{suggestions.map(([icon,label,prompt]) => <button key={icon} onClick={() => { setDraft(prompt); input.current?.focus(); }}><Icon name={icon}/>{label}</button>)}</div></div></div> : <><div className="c-conversation" role="log" aria-label="Conversation"><div className="c-message-column">{messages.map((message, index) => <div className="c-turn" key={message.id}><div className="c-user-message">{message.prompt}</div><div className="c-reply"><BrandLogo/><div className="c-response">{message.answer ? <MessageContent text={message.answer}/> : message.status === 'streaming' ? <p role="status">Waiting for Brittain 4…</p> : null}{message.error && <p className="c-error" role="alert">{message.error}</p>}{message.status === 'stopped' && <p>Reply stopped.</p>}{message.note && <p>{message.note}</p>}{message.usage && <p className="c-usage">{message.usage.total_tokens?.toLocaleString()} tokens used</p>}<div className="c-reply-actions">{message.answer && <button onClick={() => copyAnswer(message.answer)}>Copy</button>}{!busy && index === messages.length - 1 && <button onClick={e => send(e, true)}>Retry</button>}</div></div></div></div>)}<span className="sr-only" role="status">{copyNotice}</span><div ref={end}/></div></div><div className="c-bottom-composer">{composer}</div></>}
    </section>
  </main>;
}
