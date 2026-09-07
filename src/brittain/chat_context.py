"""Conversation framing and token budgeting, without loading model weights."""
from .prompts import format_prompt


class PromptTooLongError(ValueError):
    """The public character limit was exceeded before tokenization."""


def validate_messages(messages, max_chars=20_000, continuing=False):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must have a role and text content")
        if message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("message role must be system, user, or assistant")
        if not isinstance(message.get("content"), str):
            raise ValueError("message content must be text")
        total += len(message["content"])
    if total > max_chars:
        raise PromptTooLongError(f"prompt too long (limit {max_chars} characters)")
    if continuing:
        # Carrying on adds no user turn. The input is the story so far, so the
        # conversation ends with the assistant, and requiring a trailing user
        # message here rejected every continuation with a 400.
        return
    if messages[-1]["role"] != "user" or not messages[-1]["content"].strip():
        raise ValueError("the last message must be a non-empty user message")


def frame_conversation(messages, story=False):
    if story:
        # ONE TURN, deliberately. Every SFT example is a single request and a
        # single story, so the model has never seen a second <|user|> and does
        # not read turn structure: given a conversation it ignores the newest
        # instruction and keeps writing the previous story. Asked for a story
        # about a president after one about a sailor it answered "and a captain
        # who built a kingdom... I tell them about my father".
        #
        # Carrying on is a different frame rather than a longer conversation.
        # See extend_story: pretraining is almost entirely novel windows
        # continuing from the window before, which is exactly the behaviour a
        # "continue" needs, and the SFT frame cannot express it because all of
        # its examples end at <|story_end|>.
        return f"<|user|>{messages[-1]['content']}<|end_message|><|assistant|>"
    # Alpaca was trained with an optional Input field. Give it the earlier
    # conversation there and keep the latest request in Instruction.
    history = "\n\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages[:-1])
    return format_prompt(messages[-1]["content"], history)


TAGS_END = "<|end_tags|>"


def last_story(messages):
    """The most recent assistant turn that actually carries a story."""
    return next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "assistant" and (m.get("content") or "").strip()),
        None,
    )


def extend_story(story, encode=None, budget=None):
    """Frame a continuation the way pretraining framed one: a longer document.

    Trimming, when the story outgrows the window, takes from the FRONT of the
    prose and keeps the tag block. The tags are the conditioning the rest of the
    story was written to, so dropping them to keep another paragraph would let
    the continuation drift out of the genre and setting it is continuing.
    """
    head, separator, prose = story.partition(TAGS_END)
    tags_block = head + separator if separator else ""
    body = prose if separator else story
    if encode is not None and budget is not None:
        # One token is the story_start marker.
        room = budget - len(encode(tags_block)) - 1
        if room < 1:
            raise ValueError(
                "The story so far does not fit the model's context window "
                "alongside the requested reply length."
            )
        while body and len(encode(body)) > room:
            # Whole paragraphs, so the continuation never opens mid-sentence.
            _, _, body = body.partition("\n\n")
            if "\n\n" not in body and len(encode(body)) > room:
                body = body[-room * 4:]
                break
    return f"<|story_start|>{tags_block}{body.rstrip()}"


def prepare_chat_context(messages, encode, context_window, max_new_tokens,
                         story=False, frame_tokens=0, continuing=False):
    """Drop oldest complete turns; never silently slice the newest instruction.

    Reserve generation space and keep system instructions. Fail clearly when
    the latest request plus instructions alone cannot fit.
    """
    # Whether there is anything to continue decides how the messages are
    # validated, so it has to be settled first: a continuation legitimately ends
    # with the assistant, and a fallback to a fresh request legitimately does not.
    previous = last_story(messages) if (story and continuing) else None
    validate_messages(messages, continuing=previous is not None)
    budget = context_window - frame_tokens - max_new_tokens
    if story:
        if previous is not None:
            prompt = extend_story(previous, encode, budget)
            continued = True
        else:
            # A fresh request, framed as the single turn the model was tuned on.
            # Nothing to continue falls here too, which is right: it becomes an
            # ordinary request rather than a continuation of nothing.
            prompt = frame_conversation(messages, story=True)
            continued = False
        prompt_tokens = len(encode(prompt))
        if prompt_tokens > budget:
            raise ValueError(
                f"This message is too long for the model's {context_window}-token "
                "context window. Shorten it or request fewer output tokens."
            )
        return prompt, {
            "window": context_window,
            "prompt_tokens": prompt_tokens + frame_tokens,
            "reserved_tokens": max_new_tokens,
            # Earlier turns are not in the prompt, but they were not dropped for
            # want of room: this model is single-turn by construction. Reported
            # separately so a client does not warn about a context overflow that
            # did not happen.
            "dropped_messages": 0,
            "single_turn": not continued,
            "continued": continued,
        }
    system = [m for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    while True:
        retained = system + turns
        prompt = frame_conversation(retained, story)
        prompt_tokens = len(encode(prompt))
        if prompt_tokens <= budget:
            return prompt, {
                "window": context_window,
                "prompt_tokens": prompt_tokens + frame_tokens,
                "reserved_tokens": max_new_tokens,
                "dropped_messages": len(messages) - len(retained),
            }
        next_user = next((i for i, m in enumerate(turns[1:], 1)
                          if m["role"] == "user"), None)
        if next_user is None:
            raise ValueError(
                f"This message is too long for the model's {context_window}-token "
                "context window. Shorten it or request fewer output tokens."
            )
        turns = turns[next_user:]
