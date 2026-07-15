"""
The instruction template (Alpaca format). Shared by prepare_sft.py (to build
training examples) and chat.py (to wrap user input the same way at inference),
so the format the model is trained on and the format it's prompted with can
never diverge.
"""

PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def format_prompt(instruction, inp=""):
    """Everything up to and including '### Response:\\n' — the part the model
    reads as context. The response it generates comes after."""
    if inp and inp.strip():
        return PROMPT_WITH_INPUT.format(instruction=instruction.strip(), input=inp.strip())
    return PROMPT_NO_INPUT.format(instruction=instruction.strip())
