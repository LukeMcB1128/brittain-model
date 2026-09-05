"""Ask brittain-shakespeare for a story in words.

    python3 scripts/inference/story_chat.py checkpoints/shakespeare_80m_sft/weights.pt

``story.py`` frames a request the way pretraining framed a document: a
``<|story_start|>`` marker and a tag block you write yourself. SFT taught a
different frame, and the two are not interchangeable -- asking this checkpoint
for a story through the pretraining frame skips the instruction layer entirely.

    <|user|>{what you typed}<|end_message|><|assistant|>

The model answers with a tag block and then the story it wrote to it. The tags
are printed separately because they are the interesting part: they show which
constraints the model read out of the request, which is exactly where an
instruction that did not land shows up.

    /tags Genre: Tragedy      force tags instead of letting the model choose
    /free                     go back to letting it choose
    /temp X  /top_p X  /tokens N  /penalty X
    /help  /quit
"""
from __future__ import annotations

import argparse
import codecs
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.checkpoint_v3 import load_brittain3_checkpoint
from brittain.tags import parse_request, render
from brittain.tokenizer_story import StoryTokenizer

DIM, BOLD, RESET = (
    ("\033[2m", "\033[1m", "\033[0m") if sys.stdout.isatty() else ("", "", "")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--ask", default="", help="one request, then exit")
    parser.add_argument("--tags", default="", help="force the tag block")
    parser.add_argument("--max-tokens", type=int, default=1300)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.92)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_prompt(tokenizer, instruction: str, tags: dict[str, str]) -> list[int]:
    """Frame one request the way the SFT examples were framed.

    Forced tags are appended after the assistant turn, which is where the model
    would have written them itself. That leaves it completing a tag block it did
    not choose, which is the same position it is in during training whenever the
    block is partly given.
    """
    special = tokenizer.special_ids
    ids = [
        special["<|user|>"],
        *tokenizer.encode(instruction),
        special["<|end_message|>"],
        special["<|assistant|>"],
    ]
    if tags:
        ids += [
            special["<|tags|>"],
            *tokenizer.encode(render(tags)),
            special["<|end_tags|>"],
        ]
    return ids


def generate(model, tokenizer, device, ids, args) -> tuple[str, str, bool]:
    """Return (tag block, story, closed cleanly)."""
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    special = tokenizer.special_ids
    stop = {special["<|end_message|>"], tokenizer.eot, tokenizer.pad}
    tags_end = special["<|end_tags|>"]
    story_end = special["<|story_end|>"]

    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    # The tags token is the only signal needed. Left to itself the model emits
    # one first and this flips on; with forced tags the prompt already closed the
    # block, so the model starts in the story and this stays off.
    in_tags = False
    tag_text, story = "", ""
    closed = False
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for token in model.stream(
            tensor, args.max_tokens,
            temperature=args.temperature, top_k=None, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        ):
            value = int(token[0, -1].item())
            if value in stop:
                closed = True
                break
            if value == story_end:
                closed = True
                break
            if value == tags_end:
                in_tags = False
                continue
            if value == special["<|tags|>"]:
                in_tags = True
                continue
            piece = utf8.decode(tokenizer.token_bytes(value))
            if in_tags:
                tag_text += piece
            else:
                story += piece
                print(piece, end="", flush=True)
    print()
    return tag_text.strip(), story.strip(), closed


def run(model, tokenizer, device, instruction, tags, args) -> None:
    ids = build_prompt(tokenizer, instruction, tags)
    if tags:
        print(f"{DIM}tags (forced): {render(tags)}{RESET}")
    chosen, story, closed = generate(model, tokenizer, device, ids, args)
    if chosen and not tags:
        print(f"{DIM}tags the model chose: {chosen}{RESET}")
    if not closed:
        print(f"{DIM}[hit the {args.max_tokens}-token budget without finishing]{RESET}")
    elif not story:
        print(f"{DIM}[the model produced no story]{RESET}")


def main():
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = torch.device(args.device)

    path = project_path(args.checkpoint)
    model, checkpoint = load_brittain3_checkpoint(path, device=device)
    model.to(device).eval()
    tokenizer = StoryTokenizer(
        project_path(args.tokenizer or checkpoint.get("tokenizer_path"))
    )
    if "<|user|>" not in tokenizer.special_ids:
        raise SystemExit("this tokenizer has no chat tokens")
    state = checkpoint.get("training_state", {})
    print(f"{BOLD}{path.name}{RESET}  {model.num_params():,} params  {device.type}"
          + (f"  update {state.get('global_update')}" if state else ""))

    try:
        tags = parse_request(args.tags) if args.tags else {}
    except ValueError as exc:
        raise SystemExit(f"bad --tags: {exc}")

    if args.ask:
        run(model, tokenizer, device, args.ask, tags, args)
        return

    print(f"{DIM}ask for a story in your own words. /help for commands.{RESET}")
    while True:
        try:
            line = input(f"\n{BOLD}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            return
        if line == "/help":
            print(__doc__)
            continue
        if line == "/free":
            tags = {}
            print(f"{DIM}the model chooses its own tags{RESET}")
            continue
        if line.startswith("/tags"):
            rest = line[len("/tags"):].strip()
            if not rest:
                print(f"{DIM}tags: {render(tags) if tags else '(model chooses)'}{RESET}")
                continue
            try:
                tags = parse_request(rest)
                print(f"{DIM}tags: {render(tags)}{RESET}")
            except ValueError as exc:
                print(f"{DIM}{exc}{RESET}")
            continue
        for command, attribute, cast in (
            ("/temp", "temperature", float), ("/top_p", "top_p", float),
            ("/tokens", "max_tokens", int), ("/penalty", "repetition_penalty", float),
        ):
            if line.startswith(command):
                try:
                    setattr(args, attribute, cast(line.split(maxsplit=1)[1]))
                    print(f"{DIM}{attribute} = {getattr(args, attribute)}{RESET}")
                except (IndexError, ValueError):
                    print(f"{DIM}usage: {command} VALUE{RESET}")
                break
        else:
            if line.startswith("/"):
                print(f"{DIM}unknown command; /help{RESET}")
                continue
            run(model, tokenizer, device, line, tags, args)


if __name__ == "__main__":
    main()
