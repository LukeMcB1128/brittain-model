"""Add a Continue affordance to the site: a flag on the request and a button."""
import pathlib

p = pathlib.Path(r"C:\Coding\brittain-model\site\src\App.jsx")
s = p.read_text(encoding="utf-8")

# 1. send() gains a continuing mode. A continuation sends no new user text.
old = """  async function send(retry = false) {
    if (busy || abort.current || !selectedModel) return;
    const text = draft.trim();
    if (!retry && !text) return;
    if (!retry && draft.length > 20000) {"""
new = """  async function send(retry = false, continuing = false) {
    if (busy || abort.current || !selectedModel) return;
    const text = draft.trim();
    if (!retry && !continuing && !text) return;
    if (!retry && !continuing && draft.length > 20000) {"""
assert s.count(old) == 1
s = s.replace(old, new)

old = """    const history = retry
      ? chat.messages.slice(
          0,
          chat.messages.findLastIndex((m) => m.role === "user") + 1,
        )
      : ["""
new = """    const history = retry
      ? chat.messages.slice(
          0,
          chat.messages.findLastIndex((m) => m.role === "user") + 1,
        )
      : continuing
        ? // Carrying on adds no user turn: the server reframes the story so far
          // as a longer document, which is the shape pretraining taught.
          chat.messages
      : ["""
assert s.count(old) == 1
s = s.replace(old, new)

old = """    if (!retry) setDraft("");"""
new = """    if (!retry && !continuing) setDraft("");"""
assert s.count(old) == 1
s = s.replace(old, new)

# 2. The flag itself.
old = """            model: selectedName,
            stream: true,
            raw,"""
new = """            model: selectedName,
            stream: true,
            raw,
            // An intent, not something the server should infer from wording:
            // sniffing for "continue" would misfire on a story about
            // continuing and miss "keep going".
            ...(continuing ? { continue: true } : {}),"""
assert s.count(old) == 1
s = s.replace(old, new)

# 3. The title for a continuation must not become "".
old = """      title: chat.messages.length ? chat.title : text.slice(0, 52),"""
new = """      title: chat.messages.length ? chat.title : text.slice(0, 52) || "New chat","""
assert s.count(old) == 1
s = s.replace(old, new)

# 4. Do not warn about a context overflow that did not happen.
old = """      if (recent.length < history.length || context?.dropped_messages > 0)
        setNotice("""
new = """      // single_turn is by design for the story model, not budget pressure, so it
      // must not raise the context-overflow warning.
      if (
        !context?.single_turn &&
        (recent.length < history.length || context?.dropped_messages > 0)
      )
        setNotice("""
assert s.count(old) == 1
s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("send() patched")
