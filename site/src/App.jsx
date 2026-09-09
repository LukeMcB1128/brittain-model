import { useEffect, useRef, useState } from "react";
import {
  API_ORIGIN,
  apiHeaders,
  isStory,
  modelLabel,
  readNdjson,
  requestMessages,
  visibleStory,
} from "./chat";
import "./App.css";
import BrandLogo from "./BrandLogo.jsx";

const STORAGE_KEY = "brittain.chats.v1";
const uid = () => crypto.randomUUID();
const freshChat = (model, mode) => ({
  id: uid(),
  model,
  mode,
  title: "New chat",
  messages: [],
  updated: Date.now(),
});
function restoreChats() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(saved)) return [];
    return saved
      .filter(
        (c) =>
          typeof c.id === "string" &&
          typeof c.model === "string" &&
          typeof c.title === "string" &&
          Array.isArray(c.messages),
      )
      .map((c) => ({
        ...c,
        messages: c.messages
          .filter(
            (m) =>
              ["user", "assistant"].includes(m.role) &&
              typeof m.content === "string",
          )
          .map((m) => ({
            ...m,
            status: m.status === "streaming" ? "stopped" : m.status,
          })),
      }));
  } catch {
    return [];
  }
}
function Icon({ name, ...props }) {
  const paths = {
    plus: "M12 5v14M5 12h14",
    menu: "M4 6h16M4 12h16M4 18h16",
    send: "M12 19V5m-6 6 6-6 6 6",
    stop: "M7 7h10v10H7z",
    chat: "M20 11.5a8 8 0 0 1-8 8H5l-3 2 1.5-5A8 8 0 1 1 20 11.5Z",
    copy: "M9 9h11v12H9zM15 9V3H3v12h6",
    retry: "M3 10a9 9 0 1 1 2 8M3 4v6h6",
    close: "m6 6 12 12M6 18 18 6",
    info: "M12 11v6M12 7v.1M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0",
    code: "m8 6-6 6 6 6m8-12 6 6-6 6M14 3l-4 18",
    feather: "m5 19 9-9M5 19l1-7L16 2l6 6-10 10-7 1Z",
    chevron: "m6 9 6 6 6-6",
    check: "m5 12 4 4L19 6",
  };
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      {...props}
    >
      <path d={paths[name] || paths.chat} />
    </svg>
  );
}
export function MessageContent({ text }) {
  // React escapes model output, including HTML inside fenced code.
  return text.split("```").map((part, i) => {
    if (i % 2) {
      const newline = part.indexOf("\n");
      const language = newline >= 0 ? part.slice(0, newline).trim() : "";
      const code = newline >= 0 ? part.slice(newline + 1) : part;
      return (
        <div className="code-block" key={i}>
          <div className="code-label">{language || "Code"}</div>
          <pre>
            <code>{code}</code>
          </pre>
        </div>
      );
    }
    return (
      <div className="prose" key={i}>
        {part
          .split(/(`[^`\n]+`|\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g)
          .map((s, j) =>
            s.startsWith("`") ? (
              <code key={j}>{s.slice(1, -1)}</code>
            ) : s.startsWith("**") ? (
              <strong key={j}>{s.slice(2, -2)}</strong>
            ) : s.startsWith("*") ? (
              <em key={j}>{s.slice(1, -1)}</em>
            ) : (
              s
            ),
          )}
      </div>
    );
  });
}
function App() {
  const [chats, setChats] = useState(restoreChats);
  const [activeId, setActiveId] = useState(null);
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [connection, setConnection] = useState("loading");
  const [connectionError, setConnectionError] = useState("");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const [details, setDetails] = useState(false);
  const [copied, setCopied] = useState(null);
  const [storageError, setStorageError] = useState("");
  const [notice, setNotice] = useState("");
  const abort = useRef(null);
  const list = useRef(null);
  const composer = useRef(null);
  const stickToBottom = useRef(true);
  const active = chats.find((c) => c.id === activeId);
  const selectedName = active?.model || model;
  const selectedModel = models.find((m) => m.name === selectedName);
  const raw = (selectedModel?.mode ?? active?.mode) === "raw";
  const messages = active?.messages || [];
  const story = isStory(selectedModel || { name: selectedName });

  async function loadModels(signal) {
    setConnection("loading");
    setConnectionError("");
    try {
      const response = await fetch(`${API_ORIGIN}/api/tags`, {
        headers: apiHeaders,
        signal: signal || AbortSignal.timeout(15000),
      });
      if (!response.ok)
        throw new Error(`Server responded with ${response.status}.`);
      const data = await response.json();
      const available = (data.models || []).filter(
        (m) => typeof m.name === "string",
      );
      if (!available.length)
        throw new Error("No models are currently available.");
      setModels(available);
      setModel((current) =>
        available.some((m) => m.name === current) ? current : available[0].name,
      );
      setConnection("online");
    } catch (err) {
      if (err.name === "AbortError") return;
      setConnection("offline");
      setConnectionError(
        `Cannot reach the model server. ${err.message}`,
      );
    }
  }
  useEffect(() => {
    const controller = new AbortController();
    loadModels(
      AbortSignal.any([controller.signal, AbortSignal.timeout(15000)]),
    );
    return () => {
      controller.abort();
      abort.current?.abort();
    };
  }, []);
  useEffect(() => {
    if (busy) return;
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(chats.filter((c) => c.messages.length)),
      );
      setStorageError("");
    } catch {
      setStorageError(
        "Browser storage is unavailable or full. This session’s changes may not be saved.",
      );
    }
  }, [chats, busy]);
  useEffect(() => {
    if (stickToBottom.current && list.current)
      list.current.scrollTop = list.current.scrollHeight;
  }, [chats, busy, activeId]);
  useEffect(() => {
    if (composer.current) {
      composer.current.style.height = "auto";
      composer.current.style.height = `${Math.min(composer.current.scrollHeight, 180)}px`;
    }
  }, [draft]);
  function startChat(nextModel = selectedName) {
    if (busy) return;
    setModel(nextModel);
    setActiveId(null);
    setDraft("");
    setNotice("");
    setSidebar(false);
    stickToBottom.current = true;
    composer.current?.focus();
  }
  function openChat(chat) {
    setActiveId(chat.id);
    setModel(chat.model);
    setDraft("");
    setNotice("");
    setSidebar(false);
    stickToBottom.current = true;
  }
  function updateMessage(chatId, messageId, patch) {
    setChats((current) =>
      current.map((c) =>
        c.id === chatId
          ? {
              ...c,
              messages: c.messages.map((m) =>
                m.id === messageId ? { ...m, ...patch } : m,
              ),
            }
          : c,
      ),
    );
  }
  async function send(retry = false, continuing = false) {
    if (busy || abort.current || !selectedModel) return;
    const text = draft.trim();
    if (!retry && !continuing && !text) return;
    if (!retry && !continuing && draft.length > 20000) {
      setNotice("Please shorten your message to 20,000 characters or fewer.");
      return;
    }
    const chat = active || freshChat(selectedName, selectedModel.mode);
    const history = retry
      ? chat.messages.slice(
          0,
          chat.messages.findLastIndex((m) => m.role === "user") + 1,
        )
      : continuing
        ? // Carrying on adds no user turn: the server reframes the story so far
          // as a longer document, which is the shape pretraining taught.
          chat.messages
      : [
          ...chat.messages,
          { id: uid(), role: "user", content: raw ? draft : text },
        ];
    if (!history.length) return;
    const reply = {
      id: uid(),
      role: "assistant",
      content: "",
      status: "streaming",
    };
    const updated = {
      ...chat,
      title: chat.messages.length ? chat.title : text.slice(0, 52) || "New chat",
      messages: [...history, reply],
      updated: Date.now(),
    };
    setChats((current) => [
      updated,
      ...current.filter((c) => c.id !== chat.id),
    ]);
    setActiveId(chat.id);
    if (!retry && !continuing) setDraft("");
    setBusy(true);
    setNotice("");
    stickToBottom.current = true;
    const controller = new AbortController();
    abort.current = controller;
    let content = "";
    let context = null;
    const recent = requestMessages(history);
    try {
      const response = await fetch(
        `${API_ORIGIN}/api/${raw ? "generate" : "chat"}`,
        {
          method: "POST",
          signal: controller.signal,
          headers: { ...apiHeaders, "Content-Type": "application/json" },
          body: JSON.stringify({
            model: selectedName,
            stream: true,
            raw,
            // An intent, not something the server should infer from wording:
            // sniffing for "continue" would misfire on a story about
            // continuing and miss "keep going".
            ...(continuing ? { continue: true } : {}),
            ...(raw
              ? { prompt: history.at(-1).content }
              : { messages: recent }),
            options: {
              // A default is a suggestion and a max is a limit, so taking the
              // smaller of the two let the default win every time: the server
              // advertises max_tokens 1500 for the story model and this asked
              // for 256, cutting stories off mid-word. The tag block the model
              // emits before the prose comes out of the same budget, so the
              // story got barely 190 tokens.
              //
              // num_predict is a ceiling, not a target. Generation stops at
              // <|end_message|> whenever the model is finished, so asking for
              // the whole allowance costs nothing when it ends early. Raw
              // completion keeps the small default: autocomplete wants a line,
              // not an essay, and there is no stop token to end it.
              num_predict: raw
                ? (selectedModel.defaults?.num_predict ?? 64)
                : (selectedModel.max_tokens || 512),
              temperature:
                selectedModel.defaults?.temperature ?? (raw ? 0.2 : 0.5),
            },
          }),
        },
      );
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(
          data.error || `Server responded with ${response.status}.`,
        );
      }
      await readNdjson(response, (chunk) => {
        content += chunk.message?.content ?? chunk.response ?? "";
        if (chunk.context) context = chunk.context;
        updateMessage(chat.id, reply.id, { content });
      });
      updateMessage(chat.id, reply.id, { content, status: "done", context });
      // single_turn is by design for the story model, not budget pressure, so it
      // must not raise the context-overflow warning.
      if (
        !context?.single_turn &&
        (recent.length < history.length || context?.dropped_messages > 0)
      )
        setNotice(
          "Older messages were left out to fit this model’s context window. Your full conversation is still saved here.",
        );
    } catch (err) {
      updateMessage(chat.id, reply.id, {
        content,
        status: controller.signal.aborted ? "stopped" : "error",
        error: controller.signal.aborted ? "" : err.message,
      });
    } finally {
      abort.current = null;
      setBusy(false);
      composer.current?.focus();
    }
  }
  async function copyMessage(message) {
    try {
      await navigator.clipboard.writeText(
        story
          ? visibleStory(message.content, message.status !== "streaming")
          : message.content,
      );
      setCopied(message.id);
      setTimeout(() => setCopied(null), 1800);
    } catch {
      setNotice("Could not copy. You can select and copy the reply directly.");
    }
  }
  const suggestions = raw
    ? [
        ["Continue a function", "def fibonacci(n):\n    "],
        ["Complete a class", "class Stack:\n    def __init__(self):\n        "],
      ]
    : story
      ? [
          [
            "A story worth telling",
            "Write a short story about a lighthouse keeper who receives a letter from the sea.",
          ],
          [
            "Something unexpected",
            "Write a gentle fable about a fox who is afraid of the dark.",
          ],
        ]
      : [
          [
            "Build something small",
            "Write a Python function that checks whether a string is a palindrome.",
          ],
          [
            "Work through a problem",
            "Explain how binary search works, with a Python example.",
          ],
        ];
  return (
    <div className="chat-app">
      {sidebar && (
        <button
          className="sidebar-scrim"
          aria-label="Close sidebar"
          onClick={() => setSidebar(false)}
        />
      )}
      <aside
        className={`sidebar ${sidebar ? "is-open" : ""}`}
        aria-label="Conversations"
      >
        <div className="brand">
          <BrandLogo/>
          <span>
            BRITTAIN
            <span className="brand-caption">
              Experimental Models
            </span>
          </span>
          <button
            className="icon-button mobile-only"
            aria-label="Close sidebar"
            onClick={() => setSidebar(false)}
          >
            <Icon name="close" />
          </button>
        </div>
        <button
          className="new-chat"
          onClick={() => startChat()}
          disabled={busy}
        >
          <Icon name="plus" />
          New chat<span className="new-chat-symbol">↗</span>
        </button>
        <div className="sidebar-label">YOUR CONVERSATIONS</div>
        <nav className="chat-history" aria-label="Chat history">
          {!chats.some((c) => c.messages.length) && (
            <p className="history-empty">
              Your conversations will appear here.
            </p>
          )}
          {chats
            .filter((c) => c.messages.length)
            .map((chat) => (
              <button
                key={chat.id}
                className={`history-item ${chat.id === activeId ? "active" : ""}`}
                onClick={() => openChat(chat)}
                disabled={busy}
                aria-current={chat.id === activeId ? "page" : undefined}
              >
                <Icon name="chat" />
                <span>
                  {chat.title}
                  <small>{modelLabel(chat.model)}</small>
                </span>
              </button>
            ))}
        </nav>
        <div className="sidebar-footer">
          <span className="avatar">B</span>
          <div>
            Built from scratch<small>Chats saved in this browser</small>
          </div>
          <a
            href="https://github.com/lukemcb1128/brittain-model"
            target="_blank"
            rel="noreferrer"
            aria-label="BRITTAIN on GitHub"
          >
            ↗
          </a>
        </div>
      </aside>
      <main className="main-chat">
        <header className="topbar">
          <button
            className="icon-button mobile-only"
            aria-label="Open sidebar"
            aria-expanded={sidebar}
            onClick={() => setSidebar(true)}
          >
            <Icon name="menu" />
          </button>
          <div className="model-picker">
            <label className="sr-only" htmlFor="model">
              Model
            </label>
            <select
              id="model"
              value={selectedName}
              onChange={(e) => startChat(e.target.value)}
              disabled={busy || !models.length}
            >
              {!models.length && (
                <option value="">
                  {connection === "loading"
                    ? "Connecting…"
                    : "No models available"}
                </option>
              )}
              {active && !selectedModel && (
                <option value={active.model}>
                  {modelLabel(active.model)} · unavailable
                </option>
              )}
              {models.map((m) => (
                <option key={m.name} value={m.name}>
                  {modelLabel(m.name)} · {m.details?.parameter_size || m.name}
                </option>
              ))}
            </select>
            <Icon name="chevron" />
          </div>
          <span className="model-mode">{raw ? "Completion" : "Chat"}</span>
          <div className="topbar-right">
            <span className={`connection ${connection}`}>
              <i />
              {connection === "online"
                ? "Connected"
                : connection === "loading"
                  ? "Connecting"
                  : "Offline"}
            </span>
            <button
              className={`icon-button ${details ? "selected" : ""}`}
              aria-label="Model details"
              aria-expanded={details}
              onClick={() => setDetails(!details)}
            >
              <Icon name="info" />
            </button>
          </div>
        </header>
        {details && (
          <section className="model-details" aria-label="Model details">
            <strong>{selectedName || "No model selected"}</strong>
            <span>
              {selectedModel?.details?.parameter_size || "—"} parameters ·{" "}
              {Number(selectedModel?.context || 0).toLocaleString()} token
              context · {selectedModel?.mode || "—"}
            </span>
            <span>
              Languages:{" "}
              {selectedModel?.languages?.join(", ") ||
                selectedModel?.details?.languages ||
                "Not specified"}{" "}
              · FIM:{" "}
              {selectedModel?.supports_fim ? "Supported" : "Not supported"}
            </span>
          </section>
        )}
        <div
          className="conversation-scroll"
          ref={list}
          onScroll={() => {
            const el = list.current;
            stickToBottom.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 100;
          }}
        >
          {messages.length === 0 ? (
            <section className="welcome">
              <BrandLogo/>
              <div className="eyebrow">
                BRITTAIN / {raw ? "CODE" : story ? "STORIES" : "CHAT"}
              </div>
              <h1>
                {raw
                  ? "Pick up where you left off."
                  : story
                    ? "Where should the story begin?"
                    : "What are we building today?"}
              </h1>
              <p>
                {raw
                  ? "Share a code prefix and let Coder XS continue it."
                  : "Start with a thought. See where it takes you."}
              </p>
              <div className="suggestions">
                {suggestions.map(([title, prompt]) => (
                  <button
                    key={title}
                    onClick={() => {
                      setDraft(prompt);
                      composer.current?.focus();
                    }}
                  >
                    <Icon name={story ? "feather" : "code"} />
                    <span>
                      {title}
                      <small>{prompt}</small>
                    </span>
                    <span className="suggestion-arrow">↗</span>
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div
              className="messages"
              role="log"
              aria-label="Conversation"
              aria-live="off"
            >
              {messages.map((message, index) => {
                // A reply that has stopped arriving is judged differently
                // from one still streaming: an unclosed tag block is normal
                // mid-stream and means the model derailed once it is over.
                const text =
                  message.role === "assistant" && story
                    ? visibleStory(
                        message.content,
                        message.status !== "streaming",
                      )
                    : message.content;
                return (
                  <article
                    className={`message ${message.role}`}
                    key={message.id}
                    aria-label={
                      message.role === "user" ? "You" : modelLabel(selectedName)
                    }
                  >
                    {message.role === "assistant" && (
                      <div className="assistant-heading">
                        <BrandLogo/>
                        <strong>{modelLabel(selectedName)}</strong>
                      </div>
                    )}
                    <div className="message-body">
                      {message.role === "user" ? (
                        <div className="user-bubble">{text}</div>
                      ) : (
                        <>
                          {text ? (
                            raw ||
                            (!story &&
                              !text.includes("```") &&
                              /^\s*(?:def |class |import |from \S+ import |async def |function |const |let |var |#include|public class |package |fn |use )/.test(
                                text,
                              )) ? (
                              <div className="code-block">
                                <div className="code-label">
                                  {raw ? "Completion" : "Code"}
                                </div>
                                <pre>
                                  <code>{text}</code>
                                </pre>
                              </div>
                            ) : (
                              <MessageContent text={text} />
                            )
                          ) : message.status === "streaming" ? (
                            <div className="thinking" role="status">
                              <span />
                              <span />
                              <span />
                              <span className="sr-only">Writing a reply</span>
                            </div>
                          ) : (
                            <p className="muted">
                              {message.status === "stopped"
                                ? "Reply stopped."
                                : message.status === "error"
                                  ? "Could not finish this reply."
                                  : "The model returned no visible text. Try again."}
                            </p>
                          )}
                          {message.status === "error" && (
                            <p className="message-error" role="alert">
                              {message.error}
                            </p>
                          )}
                          {message.status === "stopped" && text && (
                            <small className="muted">Stopped</small>
                          )}
                          {message.status !== "streaming" && (
                            <div className="message-actions">
                              {text && (
                                <button
                                  className="icon-button"
                                  aria-label={
                                    copied === message.id
                                      ? "Copied"
                                      : "Copy reply"
                                  }
                                  onClick={() => copyMessage(message)}
                                >
                                  <Icon
                                    name={
                                      copied === message.id ? "check" : "copy"
                                    }
                                  />
                                </button>
                              )}
                              {index === messages.length - 1 && (
                                <button
                                  className="retry-button"
                                  disabled={busy || !selectedModel}
                                  onClick={() => send(true)}
                                >
                                  <Icon name="retry" />
                                  Try again
                                </button>
                              )}
                              {/* Story models cannot be asked to carry on in
                                  words -- they were tuned on one request and
                                  one story, so a second turn reads as a new
                                  request. The button says "continue" to the
                                  server, which reframes the story so far as a
                                  longer document instead. */}
                              {index === messages.length - 1 &&
                                story &&
                                text && (
                                  <button
                                    className="retry-button"
                                    disabled={busy || !selectedModel}
                                    onClick={() => send(false, true)}
                                  >
                                    <Icon name="feather" />
                                    Continue
                                  </button>
                                )}
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </div>
        <div className="composer-dock">
          {(connectionError ||
            (active && connection === "online" && !selectedModel)) && (
            <div className="connection-error" role="alert">
              {connectionError ||
                "This chat’s model is not currently served. Choose another model to start a new chat."}
              <button
                onClick={() => loadModels()}
                disabled={connection === "loading"}
              >
                Reconnect
              </button>
            </div>
          )}
          {(notice || storageError) && (
            <p className="notice" role="status">
              {storageError || notice}
            </p>
          )}
          {raw && (
            <p className="completion-note">
              Code completion · Each message continues your code independently.
            </p>
          )}
          {selectedModel && !raw && !selectedModel.chat_history && (
            <p className="completion-note">
              Conversation memory isn’t enabled on this server yet. Replies use
              only your latest message.
            </p>
          )}
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              send();
            }}
          >
            <label htmlFor="message" className="sr-only">
              {raw ? "Code to continue" : "Message"}
            </label>
            <textarea
              id="message"
              ref={composer}
              rows="1"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder={
                raw
                  ? "Paste code to continue…"
                  : `Message ${modelLabel(selectedName)}…`
              }
              onKeyDown={(e) => {
                if (
                  e.key === "Enter" &&
                  !e.shiftKey &&
                  !e.nativeEvent.isComposing
                ) {
                  e.preventDefault();
                  if (!busy) send();
                }
              }}
            />
            <div className="composer-bottom">
              <span>
                <Icon name={story ? "feather" : "code"} />
                {selectedModel
                  ? `${Number(selectedModel.context).toLocaleString()} context`
                  : "Waiting for a model"}
              </span>
              {busy ? (
                <button
                  type="button"
                  className="send-button stop-button"
                  aria-label="Stop generating"
                  onClick={() => abort.current?.abort()}
                >
                  <Icon name="stop" />
                </button>
              ) : (
                <button
                  type="submit"
                  className="send-button"
                  aria-label="Send message"
                  disabled={!draft.trim() || !selectedModel}
                >
                  <Icon name="send" />
                </button>
              )}
            </div>
          </form>
          <div className="composer-caption">
            <span>
              Small, experimental models. Site created by Brittain.
            </span>
            <span>Enter to send · Shift + Enter for a new line</span>
          </div>
          <span className="sr-only" role="status">
            {busy
              ? "Generating reply"
              : messages.length
                ? "Ready for your next message"
                : ""}
          </span>
        </div>
      </main>
    </div>
  );
}
export default App;
