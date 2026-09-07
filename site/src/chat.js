export const API_ORIGIN = (
  import.meta.env?.VITE_API_ORIGIN ||
  "https://fragility-devoutly-dazzling.ngrok-free.dev"
).replace(/\/$/, "");
export const apiHeaders = { "ngrok-skip-browser-warning": "true" };

const TAGS_OPEN = "<|tags|>";
const TAGS_END = "<|end_tags|>";

// Accumulated text, not individual chunks: partial tags must never flash onscreen.
//
// `done` matters because a half-arrived tag block and one the model never closed
// look identical, and the right answer differs. Hiding everything after <|tags|>
// until <|end_tags|> arrives was correct while streaming and lost the whole reply
// when the block was never closed — the model sometimes drifts out of the tags
// straight into prose ("[Genre: my partner in this ridiculous charade"), and
// roughly one continuation in eight came back blank because 2,000 characters of
// story were swallowed by the fallback that was only meant to hide a partial tag.
export function visibleStory(text, done = false) {
  // The block is found wherever it is, not assumed to start the string: the
  // wire can open with <|assistant|>, and anchoring at position 0 would leave
  // the tag values on screen as ordinary prose.
  const open = text.indexOf(TAGS_OPEN);
  let body = text;
  if (open !== -1) {
    const before = text.slice(0, open);
    const rest = text.slice(open + TAGS_OPEN.length);
    const end = rest.indexOf(TAGS_END);
    if (end !== -1) {
      body = before + rest.slice(end + TAGS_END.length);
    } else if (!done) {
      body = before;
    } else {
      // Finished without ever closing the block. Drop the complete [Key: Value]
      // pairs and the unterminated one it broke off in, and keep the prose.
      body =
        before +
        rest
          .replace(/^(?:\s*\[[^[\]]*\])*/, "")
          .replace(/^\s*\[[A-Za-z][A-Za-z -]*:\s*/, "");
    }
  }
  return body
    .replace(/<\|[^|]*\|>/g, "")
    .replace(/<(?:\|[^>]*)?$/, "")
    .trimStart();
}
export function isStory(model) {
  return (
    model?.details?.tokenizer === "brittain_shakespeare_bpe" ||
    /shakespeare/i.test(model?.name || "")
  );
}
export function modelLabel(name = "") {
  if (/shakespeare/i.test(name)) return "Shakespeare";
  if (/xs-coder/i.test(name)) return "Coder XS";
  if (/coder/i.test(name)) return "Coder";
  return name.split(":")[0] || "Choose a model";
}
// Exact token budgeting happens on the server. Preserve whole turns here while
// keeping the HTTP request inside its character limit.
export function requestMessages(messages) {
  const result = messages
    .filter((m) => m.role === "user" || (m.content && m.status !== "error"))
    .map(({ role, content }) => ({ role, content }));
  while (
    result.reduce((n, m) => n + m.content.length, 0) > 20000 &&
    result.length > 1
  ) {
    result.shift();
    while (result.length > 1 && result[0].role !== "user") result.shift();
  }
  return result;
}
export async function readNdjson(response, onChunk) {
  if (!response.body)
    throw new Error("The server returned no response stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  let complete = false;
  const consume = (line) => {
    if (!line.trim()) return;
    const chunk = JSON.parse(line);
    if (chunk.error) throw new Error(chunk.error);
    onChunk(chunk);
    if (chunk.done) complete = true;
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const lines = pending.split("\n");
      pending = lines.pop();
      lines.forEach(consume);
      if (done || complete) break;
    }
    if (pending.trim() && !complete) consume(pending);
    if (!complete)
      throw new Error(
        "The connection ended before the reply finished. Try again.",
      );
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}
