import { lazy, Suspense, useEffect, useState } from 'react';
import './public.css';
import BrandLogo from './BrandLogo.jsx';
import { AccountSettings, AuthPage } from './AccountPage.jsx';
import { getAccountSession } from './auth-client.js';

const Chat = lazy(() => import('./ChatPage.jsx'));
const ExperimentalChat = lazy(() => import('./App.jsx'));

if (window.location.hash.startsWith('#/')) {
  window.history.replaceState(null, '', window.location.hash.slice(1));
}
const path = () => `${window.location.pathname}${window.location.search}`;
const Arrow = () => <span aria-hidden="true">↗</span>;
function Link({ to, children, className = '', ...props }) { return <a href={to} className={className} {...props}>{children}</a>; }
function Brand() { return <Link to="/" className="brand-link" aria-label="Brittain home"><BrandLogo/><span>{import.meta.env.MODE === 'staging' ? 'BRITTAIN / TEST SITE' : 'BRITTAIN'}</span></Link>; }
function Header({ route, session, sessionLoading }) {
  return <><a className="skip-link" href="#main-content">Skip to main content</a><header className="site-header"><Brand/><nav aria-label="Main navigation">{[['/', 'Home'], ['/chat', 'Chat'], ['/models', 'Models']].map(([to, label]) => <Link key={to} to={to} aria-current={(route === to || (to === '/chat' && route.startsWith('/chat/')) || (to === '/models' && route.startsWith('/models/'))) ? 'page' : undefined}>{label}</Link>)}</nav>{session ? <Link to="/account" className="button compact">{session.user.name}</Link> : sessionLoading ? <span className="header-account-loading">Checking account…</span> : route === '/signup' ? <Link to="/login" className="button compact">Sign in <Arrow/></Link> : <Link to="/signup" className="button compact">Sign up free <Arrow/></Link>}</header></>;
}
function Footer() { return <footer className="site-footer"><Brand/><span>Independent models. Practical use.</span><a href="https://github.com/lukemcb1128/brittain-model" target="_blank" rel="noreferrer">GitHub <Arrow/></a><Link to="/models">Model information</Link><Link to="/privacy">Privacy</Link></footer>; }
function Home({ session }) {
  return <><main className="site-main"><section className="home-hero"><div className="hero-copy"><p className="eyebrow">BRITTAIN 4 / PUBLIC DEMO</p><h1>Meet<br/><span>Brittain 4.</span></h1><p className="lead">A 9B dense model for writing, code, and everyday questions. Use it in your browser or choose a download to run locally.</p><div className="actions"><Link to={session ? '/chat' : '/signup'} className="button primary">Try Brittain 4 <Arrow/></Link><Link to="/models" className="button">Explore models <Arrow/></Link></div><p className="secondary">Web chat requires a free Brittain account.</p></div><div className="model-sheet"><div className="sheet-heading"><span>MODEL PROFILE</span><span>04</span></div><div className="sheet-title">Brittain <span>4</span></div><dl><div><dt>Architecture</dt><dd>Dense</dd></div><div><dt>Parameters</dt><dd>9 billion</dd></div><div><dt>Model context</dt><dd>Up to 262k</dd></div><div><dt>Web chat context</dt><dd>32k</dd></div></dl><Link to="/models/brittain-4" className="text-link">Read model details <Arrow/></Link></div></section><section className="access-section"><div className="section-heading"><p className="eyebrow">HOW TO USE IT</p><h2>Two ways to work with Brittain.</h2></div><div className="access-grid"><article><span className="section-number">01 / IN YOUR BROWSER</span><h3>Start with a conversation.</h3><p>Open chat, ask a question, and continue from the reply. A free account helps us share server capacity between users.</p><Link to="/chat" className="text-link">Open web chat <Arrow/></Link><div className="access-note">No local setup · Shared capacity</div></article><article><span className="section-number">02 / ON YOUR HARDWARE</span><h3>Choose a model to run locally.</h3><p>Compare release files and quantized versions. Available context depends on the model version, runtime, and memory.</p><Link to="/models" className="text-link">Explore downloads <Arrow/></Link><div className="access-note">Your runtime · Your hardware</div></article></div></section><section className="release-note"><div><p className="eyebrow">THE MODEL COLLECTION</p><h2>Brittain 4, with room for what came before.</h2></div><p>Find the new release family in Models. Earlier experimental models have a separate area, with their own notes and downloads.</p><Link to="/models" className="text-link">View the collection <Arrow/></Link></section></main><Footer/></>;
}
function Models() {
  const [group, setGroup] = useState('brittain');
  function moveTab(event) {
    const order = ['brittain', 'experimental'];
    let index = order.indexOf(group);
    if (event.key === 'ArrowRight') index = (index + 1) % order.length;
    else if (event.key === 'ArrowLeft') index = (index - 1 + order.length) % order.length;
    else if (event.key === 'Home') index = 0;
    else if (event.key === 'End') index = order.length - 1;
    else return;
    event.preventDefault();
    const next = order[index];
    setGroup(next);
    requestAnimationFrame(() => document.getElementById(`${next}-tab`)?.focus());
  }
  const tabProps = name => ({
    id: `${name}-tab`,
    role: 'tab',
    tabIndex: group === name ? 0 : -1,
    'aria-selected': group === name,
    'aria-controls': `${name}-panel`,
    onClick: () => setGroup(name),
    onKeyDown: moveTab,
  });
  return <><main className="site-main catalog"><p className="eyebrow">MODEL COLLECTION</p><h1>Find your Brittain model.</h1><p className="lead">Release information and local downloads, in one place.</p><div className="segmented" role="tablist" aria-label="Model collection"><button {...tabProps('brittain')}>Brittain 4</button><button {...tabProps('experimental')}>Experimental models</button></div>{group === 'brittain' ? <section id="brittain-panel" role="tabpanel" aria-labelledby="brittain-tab" className="catalog-entry"><div><span className="tag">RELEASE IN PREPARATION</span><h2>Brittain 4</h2><p>A 9B dense model with support for up to 262k context. Release files and quantized versions will be listed here when confirmed.</p><Link to="/models/brittain-4" className="button">Model details <Arrow/></Link></div><dl className="entry-facts"><div><dt>Parameters</dt><dd>9B</dd></div><div><dt>Architecture</dt><dd>Dense</dd></div><div><dt>Maximum context</dt><dd>262k</dd></div></dl></section> : <section id="experimental-panel" role="tabpanel" aria-labelledby="experimental-tab" className="experimental-section"><h2>Experimental models</h2><p className="lead">Earlier work in code completion and story generation. These models are separate from the Brittain 4 family.</p>{[['Coder', 'Code and instruction experiments.'], ['Coder XS', 'Small models for code completion.'], ['Shakespeare', 'Models for story generation.']].map(([name, desc]) => <article className="experimental-row" key={name}><div><h3>{name}</h3><p>{desc}</p></div><a className="text-link" href="https://github.com/lukemcb1128/brittain-model" target="_blank" rel="noreferrer">Repository and release notes <Arrow/></a></article>)}<p className="secondary">Direct download files will be added after the release links are checked.</p></section>}<aside className="plain-note"><h3>Before you download</h3><p>Check the license, runtime support, and memory requirements for the selected file. File size alone does not describe the memory needed for long context.</p></aside></main><Footer/></>;
}
function ModelDetail() { return <><main className="site-main model-detail"><Link to="/models" className="text-link">← All models</Link><p className="eyebrow">BRITTAIN 4 / MODEL INFORMATION</p><h1>Brittain 4</h1><p className="lead">9 billion parameters. Dense architecture.<br/>Up to 262k context, where the runtime and hardware support it.</p><div className="detail-grid"><section><h2>Release files</h2><p>Downloads are being prepared. Confirmed versions, quantizations, file sizes, and setup instructions will appear here.</p><div className="file-empty"><span className="tag">NOT YET AVAILABLE</span><h3>No release files listed</h3><p>License and hardware requirements will be published with the files.</p></div></section><aside className="plain-note"><h3>Model context and web chat</h3><p>The model supports up to 262k context. The web chat uses 32k to share server capacity.</p><p>Local use of longer context depends on your runtime and available memory.</p><Link to="/chat" className="text-link">Open chat <Arrow/></Link></aside></div><section className="detail-bottom"><h2>Evaluation and limitations</h2><p>Release evaluation results and test methods are not yet published. No benchmark comparisons are claimed here.</p></section></main><Footer/></>; }
function Privacy() { return <><main className="site-main policy-page"><p className="eyebrow">PRIVACY</p><h1>How the web chat handles data.</h1><section><h2>Account data</h2><p>Brittain stores your name, email address, password credential, and signed-in sessions. Passwords are stored as secure password hashes.</p></section><section><h2>Conversations</h2><p>Saved chat text, compacted conversation memory, tool summaries, and token usage are stored with your account. Operational exchange records are also kept to diagnose failures and evaluate the model. These records use a one-way account hash.</p></section><section><h2>Files and outside services</h2><p>Images and PDF files are sent to the chat service while you use them. Their binary data is not stored in saved chat history or operational exchange records. Web tools retrieve public pages when the model uses them.</p></section><section><h2>Your controls</h2><p>You can delete a chat from the chat sidebar. You can delete your account and its saved chats from Account settings. Account deletion also removes operational exchange records linked to your account hash.</p></section><p className="secondary">This page describes the current public demo implementation. Contact information and a formal retention period must be added before a wider public launch.</p></main><Footer/></>; }
export default function PublicApp() {
  const [route, setRoute] = useState(path);
  const [session, setSession] = useState(null);
  const [sessionLoading, setSessionLoading] = useState(true);
  async function refreshSession() {
    try { setSession(await getAccountSession()); }
    catch { setSession(null); }
    finally { setSessionLoading(false); }
  }
  useEffect(() => { refreshSession(); }, []);
  useEffect(() => { const change = () => { setRoute(path()); window.scrollTo(0, 0); }; window.addEventListener('popstate', change); return () => window.removeEventListener('popstate', change); }, []);
  const current = route.split('?')[0];
  useEffect(() => {
    const title = current === '/' ? 'Brittain 4' : current.startsWith('/chat') ? 'Chat' : current === '/models' ? 'Models' : current === '/models/brittain-4' ? 'Brittain 4 model' : current === '/privacy' ? 'Privacy' : ['/signup', '/login', '/forgot-password', '/reset-password'].includes(current) ? 'Account' : current === '/account' ? 'Settings' : 'Page not found';
    document.title = `${import.meta.env.MODE === 'staging' ? '[Test site] ' : ''}${title} | BRITTAIN`;
  }, [current]);
  if (current === '/experimental-chat') return <Suspense fallback={<main className="chat-loading">Loading experimental chat…</main>}><ExperimentalChat/></Suspense>;
  const authMode = current === '/signup' ? 'signup' : current === '/login' ? 'login' : current === '/forgot-password' ? 'forgot' : current === '/reset-password' ? 'reset' : null;
  const chatMatch = current.match(/^\/chat(?:\/([^/]+))?$/);
  const chat = Boolean(chatMatch);
  const activeChat = chat && Boolean(session);
  return <div className={`release-site ${activeChat ? 'chat-route' : ''}`}>{!activeChat && <><Header route={current} session={session} sessionLoading={sessionLoading}/><span id="main-content" className="skip-target" tabIndex="-1"/></>} {current === '/' ? <Home session={session}/> : current === '/models' ? <Models/> : current === '/models/brittain-4' ? <ModelDetail/> : current === '/privacy' ? <Privacy/> : authMode ? <><AuthPage mode={authMode} onSession={refreshSession}/><Footer/></> : current === '/account' ? sessionLoading ? <main className="site-main account-loading">Loading account…</main> : session ? <><AccountSettings session={session} onSession={refreshSession}/><Footer/></> : <><AuthPage mode="login" onSession={refreshSession}/><Footer/></> : chat ? sessionLoading ? <main className="chat-loading">Loading chat…</main> : session ? <Suspense fallback={<main className="chat-loading">Loading chat…</main>}><Chat key={session.user.id} session={session} initialChatId={chatMatch[1] ? decodeURIComponent(chatMatch[1]) : ''}/></Suspense> : <><AuthPage mode="login" onSession={refreshSession} nextPath={current}/><Footer/></> : <main className="site-main catalog"><h1>Page not found.</h1><Link to="/">Return home</Link></main>}</div>;
}
