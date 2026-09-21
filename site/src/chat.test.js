import assert from "node:assert/strict";
import test from "node:test";
import { readNdjson, requestMessages, visibleStory, withoutReferenceNotes } from "./chat.js";

test("Shakespeare metadata never leaks at any stream boundary", () => {
  const wire =
    "<|assistant|><|tags|>[Voice: Modern] [Genre: Fable]<|end_tags|><|story_start|>The sea was quiet.<|story_end|><|end_message|>";
  for (let end = 1; end <= wire.length; end++) {
    const visible = visibleStory(wire.slice(0, end));
    assert.ok(
      "The sea was quiet.".startsWith(visible),
      `Leaked output at ${end}: ${visible}`,
    );
  }
  assert.equal(visibleStory(wire), "The sea was quiet.");
  assert.equal(visibleStory("A story without tags."), "A story without tags.");
  assert.equal(visibleStory("<|tags|>[Genre: unfinished"), "");
});
test("an unclosed tag block loses the tags, not the story", () => {
  // The model sometimes drifts out of the tag block straight into prose without
  // ever emitting <|end_tags|>. Hiding everything after <|tags|> until the
  // closing marker arrives is right while streaming and threw away the whole
  // reply once the response was over: roughly one continuation in eight came
  // back blank with two thousand characters of story behind it.
  const derailed =
    "<|tags|>[Voice: Modern] [Genre: my partner in this charade. Darrin sat down.";
  assert.equal(visibleStory(derailed, true), "my partner in this charade. Darrin sat down.");
  // Still arriving, so the same text is a partial tag block and shows nothing.
  assert.equal(visibleStory(derailed, false), "");
  // A closed block behaves the same either way.
  const closed = "<|tags|>[Genre: Fable]<|end_tags|>The sea was quiet.";
  assert.equal(visibleStory(closed, true), "The sea was quiet.");
  assert.equal(visibleStory(closed, false), "The sea was quiet.");
});
test("request clipping preserves the latest whole turn and excludes failed replies", () => {
  const latest = { role: "user", content: "Continue." };
  assert.deepEqual(
    requestMessages([
      { role: "user", content: "x".repeat(19900) },
      { role: "assistant", content: "x".repeat(200) },
      latest,
    ]),
    [latest],
  );
  assert.deepEqual(
    requestMessages([
      latest,
      { role: "assistant", content: "partial", status: "error" },
    ]),
    [latest],
  );
});
function responseFor(bytes, chunkSize = 1) {
  return new Response(
    new ReadableStream({
      start(controller) {
        for (let i = 0; i < bytes.length; i += chunkSize)
          controller.enqueue(bytes.slice(i, i + chunkSize));
        controller.close();
      },
    }),
  );
}
test("NDJSON handles split UTF-8, multiple records, and a final record without newline", async () => {
  const wire = new TextEncoder().encode(
    "\n" +
      JSON.stringify({ message: { content: "sea 🌊 café" } }) +
      "\n" +
      JSON.stringify({ done: true }),
  );
  for (const size of [1, 7, 10000]) {
    const chunks = [];
    await readNdjson(responseFor(wire, size), (chunk) => chunks.push(chunk));
    assert.equal(chunks[0].message.content, "sea 🌊 café");
    assert.equal(chunks[1].done, true);
  }
});
test("stream errors and premature disconnects surface instead of silently completing", async () => {
  const encode = (text) => responseFor(new TextEncoder().encode(text));
  await assert.rejects(
    readNdjson(encode('{"error":"GPU unavailable"}\n'), () => {}),
    /GPU unavailable/,
  );
  await assert.rejects(
    readNdjson(encode('{"response":"partial"}\n'), () => {}),
    /before the reply finished/,
  );
  await assert.rejects(
    readNdjson(encode("not json\n"), () => {}),
    SyntaxError,
  );
});

test("a leaked reference note never reaches the reader", () => {
  // The server tells the model what it called using a bracketed note on the
  // user side. It sometimes writes one back out; a real reply ended with
  // "[Tools you used on this turn: search_curriculum(AP Calculus BC)]".
  const leaked = "AP Calculus BC includes everything from AB." + String.fromCharCode(10, 10)
    + "[Tools you used on this turn: search_curriculum(AP Calculus BC)]";
  assert.equal(withoutReferenceNotes(leaked), "AP Calculus BC includes everything from AB.");

  const other = "[For your reference: on your previous turn you used web_search(x).]";
  assert.equal(withoutReferenceNotes(other), "");

  // Ordinary brackets are the reader's, not ours.
  const kept = "Use array[0] and see [the docs](https://example.com).";
  assert.equal(withoutReferenceNotes(kept), kept);
});
