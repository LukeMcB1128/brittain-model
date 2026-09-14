import { useEffect, useRef, useState } from 'react';
import {
  accountConfig,
  deleteAccount,
  requestPasswordReset,
  resetPassword,
  signIn,
  signOut,
  signUp,
} from './auth-client.js';

function safeNext(value) {
  return typeof value === 'string' && value.startsWith('/') && !value.startsWith('//') ? value : '/chat';
}

function Turnstile({ siteKey, onToken }) {
  const host = useRef(null);
  useEffect(() => {
    if (!siteKey) return undefined;
    let widget;
    let cancelled = false;
    const render = () => {
      if (cancelled || !host.current || !window.turnstile) return;
      widget = window.turnstile.render(host.current, {
        sitekey: siteKey,
        theme: 'dark',
        callback: onToken,
        'expired-callback': () => onToken(''),
        'error-callback': () => onToken(''),
      });
    };
    const existing = document.querySelector('script[data-brittain-turnstile]');
    if (window.turnstile) render();
    else if (existing) existing.addEventListener('load', render, { once: true });
    else {
      const script = document.createElement('script');
      script.src = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit';
      script.async = true;
      script.defer = true;
      script.dataset.brittainTurnstile = 'true';
      script.addEventListener('load', render, { once: true });
      document.head.appendChild(script);
    }
    return () => {
      cancelled = true;
      if (widget !== undefined && window.turnstile) window.turnstile.remove(widget);
    };
  }, [siteKey, onToken]);
  return siteKey ? <div className="account-turnstile" ref={host}/> : null;
}

function AccountForm({ mode, onSession }) {
  const params = new URLSearchParams(window.location.search);
  const next = safeNext(params.get('next'));
  const token = params.get('token') || '';
  const [config, setConfig] = useState({ emailVerification: false, turnstileSiteKey: '' });
  const [captchaToken, setCaptchaToken] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState(params.get('error') === 'INVALID_TOKEN' ? 'This password reset link is invalid or has expired.' : '');
  useEffect(() => { accountConfig().then(setConfig); }, []);
  async function submit(event) {
    event.preventDefault();
    setPending(true);
    setError('');
    setNotice('');
    const fields = new FormData(event.currentTarget);
    try {
      if (mode === 'signup') {
        await signUp({ name: fields.get('name'), email: fields.get('email'), password: fields.get('password'), captchaToken });
        if (config.emailVerification) {
          setNotice('Check your email and select the verification link to finish your account.');
        } else {
          await onSession();
          window.location.assign(next);
        }
      } else if (mode === 'login') {
        await signIn({ email: fields.get('email'), password: fields.get('password'), captchaToken });
        await onSession();
        window.location.assign(next);
      } else if (mode === 'forgot') {
        await requestPasswordReset(fields.get('email'), captchaToken);
        setNotice('If the address belongs to an account, a reset link is on its way.');
      } else if (mode === 'reset') {
        if (!token) throw new Error('This password reset link is invalid or has expired.');
        await resetPassword(fields.get('password'), token);
        setNotice('Your password was changed. You can now sign in.');
      }
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setPending(false);
    }
  }
  const heading = mode === 'signup' ? 'Create a free account' : mode === 'login' ? 'Sign in' : mode === 'forgot' ? 'Reset your password' : 'Choose a new password';
  return <section className="signup-panel account-panel">
    <span className="tag">BRITTAIN ACCOUNT</span>
    <h2>{heading}</h2>
    <form className="account-form" onSubmit={submit}>
      {mode === 'signup' && <label><span>Name</span><input name="name" autoComplete="name" required maxLength="80"/></label>}
      {mode !== 'reset' && <label><span>Email address</span><input name="email" type="email" autoComplete="email" required/></label>}
      {['signup', 'login', 'reset'].includes(mode) && <label><span>{mode === 'reset' ? 'New password' : 'Password'}</span><input name="password" type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength="10" maxLength="128"/>{mode !== 'login' && <small>Use at least 10 characters.</small>}</label>}
      {['signup', 'login', 'forgot'].includes(mode) && <Turnstile siteKey={config.turnstileSiteKey} onToken={setCaptchaToken}/>} 
      {error && <p className="account-error" role="alert">{error}</p>}
      {notice && <p className="account-notice" role="status">{notice}</p>}
      <button className="button primary full-width" disabled={pending || (config.turnstileSiteKey && !captchaToken)}>{pending ? 'Please wait…' : mode === 'signup' ? 'Create account' : mode === 'login' ? 'Sign in' : mode === 'forgot' ? 'Send reset link' : 'Save password'}</button>
    </form>
    <div className="account-links">
      {mode === 'signup' && <a href="/login">Already have an account? Sign in</a>}
      {mode === 'login' && <><a href="/signup">Create an account</a>{config.emailVerification && <a href="/forgot-password">Forgot password?</a>}</>}
      {['forgot', 'reset'].includes(mode) && <a href="/login">Return to sign in</a>}
    </div>
  </section>;
}

export function AuthPage({ mode, onSession }) {
  return <main className="signup-layout"><section><p className="eyebrow">BRITTAIN ACCOUNT</p><h1>Your Brittain account.</h1><p className="lead">Free web chat with Brittain 4.</p><div className="signup-benefits"><p><span>01</span> Use the hosted model without local setup.</p><p><span>02</span> Save conversations across visits.</p><p><span>03</span> Share server capacity through account limits.</p></div><a href="/models" className="text-link">Prefer local use? Explore models <span aria-hidden="true">↗</span></a></section><AccountForm mode={mode} onSession={onSession}/></main>;
}

export function AccountSettings({ session, onSession }) {
  const [password, setPassword] = useState('');
  const [pending, setPending] = useState('');
  const [error, setError] = useState('');
  async function logout() {
    setPending('logout');
    try { await signOut(); await onSession(); window.location.assign('/'); }
    catch (requestError) { setError(requestError.message); setPending(''); }
  }
  async function remove(event) {
    event.preventDefault();
    setPending('delete'); setError('');
    try { await deleteAccount(password); await onSession(); window.location.assign('/'); }
    catch (requestError) { setError(requestError.message); setPending(''); }
  }
  return <main className="site-main account-settings"><p className="eyebrow">ACCOUNT SETTINGS</p><h1>{session.user.name}</h1><p className="lead">{session.user.email}</p><section className="settings-card"><h2>Session</h2><p>Sign out on this device.</p><button className="button" onClick={logout} disabled={Boolean(pending)}>{pending === 'logout' ? 'Signing out…' : 'Sign out'}</button></section><section className="settings-card danger-card"><h2>Delete account</h2><p>This deletes your account and saved chats.</p><form className="account-form" onSubmit={remove}><label><span>Confirm your password</span><input type="password" value={password} onChange={event => setPassword(event.target.value)} required/></label>{error && <p className="account-error" role="alert">{error}</p>}<button className="button danger" disabled={Boolean(pending)}>{pending === 'delete' ? 'Deleting…' : 'Delete account'}</button></form></section></main>;
}
