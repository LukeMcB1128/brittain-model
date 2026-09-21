"""Which tool-call parser does vLLM have for Qwen3.5?

Without one, vLLM returns the <tool_call><function=...> markup as message
CONTENT and Brittain Code has to rescue it with parseRawToolCalls -- the
fallback path, explicitly "not a target" in the spec. With one, the server
returns proper OpenAI tool_calls with parsed arguments, which is what the
existing openai transport already understands.

That difference decides whether the web chat needs its own XML parser.
"""
from vllm.entrypoints.openai.tool_parsers import ToolParserManager

names = sorted(ToolParserManager.tool_parsers)
print("tool-call parsers available (%d):" % len(names))
for n in names:
    mark = "  <-- likely" if any(k in n.lower() for k in ("qwen", "hermes", "xml")) else ""
    print("   %s%s" % (n, mark))

print()
try:
    from vllm.reasoning import ReasoningParserManager
    rp = sorted(ReasoningParserManager.reasoning_parsers)
    print("reasoning parsers (%d):" % len(rp))
    for n in rp:
        mark = "  <-- likely" if "qwen" in n.lower() else ""
        print("   %s%s" % (n, mark))
except Exception as e:
    print("reasoning parsers: could not enumerate (%s)" % type(e).__name__)
