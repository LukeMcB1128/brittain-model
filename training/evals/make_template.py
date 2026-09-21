"""Build a BRITTAIN-4 chat template: Qwen's, plus a default system prompt.

WHY NOT WRITE A TEMPLATE FROM SCRATCH
Qwen's template already handles tools, vision content parts, thinking, and the
multi-step tool-response scan. All of that is verified working, and rewriting
it to add one string would put every one of those at risk for no gain. This
copies it and adds an else-branch.

WHY ONLY WHEN THE CALLER SENDS NONE
The template emits a system block only if messages[0].role == 'system'. Adding
an else-branch means a caller that supplies its own system prompt is completely
unaffected -- which matters because Brittain Code always sends systemPrompt(),
and stacking a second identity on top of it would be worse than nothing. The
default is for clients that send none, i.e. the web chat.

The prompt is deliberately positive about identity. An earlier version said
"you are not Qwen, Claude, GPT" and the model repeated the list back, putting
three competitor names into a reply where the user had asked nothing of the
sort.
"""
import argparse
import glob

# THIS WORDING WAS MEASURED, NOT COMPOSED. THERE IS NO STYLE PARAGRAPH ON
# PURPOSE -- ADDING ONE BACK BREAKS THE MODEL. Read this before editing.
#
# The draft that included response-style guidance ("lead with the conclusion,
# do not restate the question, say what you are unsure of") made the model
# refuse live-data questions outright: "I cannot provide today's exchange rate
# because I do not have real-time access", instead of searching. It also
# invented /etc/hostname's contents rather than reading the file.
#
# What the isolation runs established, each against a control:
#   - The Jinja injection is faithful. The same text sent as a caller's system
#     message scores identically, so this is prompt content, not a template
#     bug (scratchpad/ab_length.py).
#   - Length is not the cause. Filler prose of identical length scored 5/6
#     where the real text scored 3/6 (same file).
#   - No single paragraph is at fault. Alone, tools scores 10/10 on live-data
#     search, identity 8/10, style 8/10 (scratchpad/live_components.py).
#   - It is an INTERACTION between identity and style, and it is severe
#     (scratchpad/live_pairs.py, n=10 per cell):
#         identity + tools    10/10 search, 0 refusals, 3/3 restraint  <- this
#         tools + style        8/10 search, 0 refusals
#         identity + style     0/10 search, 2 refusals
#         all three            0/10 search, 4 refusals
#     Three rewordings failed to break the interaction: scoping the advice to
#     "once you have what you need", deleting the uncertainty clause, and
#     reordering style before tools (which was worst, at 6 refusals).
#
# For reference, neutral "You are a helpful assistant." scores 4/10 on live
# search, so identity + tools is not merely un-broken, it is better than a
# plain prompt. Some response guidance survives inside the tools paragraph
# ("report what you found rather than narrating what you are about to do"),
# which is the part that was actually load-bearing.
# The tools paragraph is BOUNDED to two domains, and says so positively.
# An earlier version ended "never state such a thing from memory", intending
# that to cover this machine and live data. The model read it as absolute and
# refused to write a history essay -- it called web_search instead, on the
# grounds that it could not write from memory. Both "write an essay" tasks
# failed that way; short knowledge questions did not, which is why the first
# restraint set missed it entirely (2+2, a haiku, "what does 404 mean").
# Stating what the model SHOULD answer from knowledge, then naming the two
# exceptions, fixes it: 5/5 machine, 4/4 live, 8/8 knowledge, 0 refusals
# (scratchpad/fix_overreach.py).
IDENTITY = (
    "You are BRITTAIN-4, a general-purpose assistant made by Luke Brittain. "
    "That is your name and origin; answer questions about yourself on that "
    "basis, and do not discuss your architecture or training data.\\n\\n"
    "Answer from your own knowledge by default. Use a tool when the question "
    "is about this machine -- its files, its command output, its repository "
    "-- or when the answer changes over time, such as prices, weather, or "
    "current events; check those rather than recalling them. Report what you "
    "found rather than narrating what you are about to do."
)

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
args = ap.parse_args()

src = glob.glob("/home/lukeb/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
                "snapshots/*/chat_template.jinja")[0]
tpl = open(src, encoding="utf-8").read()
original_len = len(tpl)

# Branch A: tools present. The system block is already open, so the default is
# appended to it exactly as a caller's system message would be.
a_old = """    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\\n\\n' + content }}
        {%- endif %}
    {%- endif %}"""
a_new = """    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\\n\\n' + content }}
        {%- endif %}
    {%- else %}
        {{- '\\n\\n' + brittain_identity }}
    {%- endif %}"""

# Branch B: no tools. Nothing is open, so the whole block has to be emitted.
b_old = """    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}
    {%- endif %}"""
b_new = """    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\\n' + content + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\n' + brittain_identity + '<|im_end|>\\n' }}
    {%- endif %}"""

for label, old, new in (("tools branch", a_old, a_new), ("no-tools branch", b_old, b_new)):
    if tpl.count(old) != 1:
        raise SystemExit("%s: expected exactly one match, found %d -- the "
                         "template changed and this patch is stale"
                         % (label, tpl.count(old)))
    tpl = tpl.replace(old, new)

tpl = ('{%- set brittain_identity = "' + IDENTITY + '" %}\n') + tpl

open(args.out, "w", encoding="utf-8").write(tpl)
print("source   : %s" % src)
print("original : %d chars" % original_len)
print("written  : %s (%d chars)" % (args.out, len(tpl)))
print("identity : %d chars (~%d tokens)" % (len(IDENTITY), len(IDENTITY) // 4))
