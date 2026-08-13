"""Interactive sampler for brittain-shakespeare, with the conditioning tags.

    python3 scripts/inference/story.py checkpoints/shakespeare_18m_pilot/best.pt

Inside the prompt:

    /tags Tense: Present, Setting: Sea     set the tag block
    /tags                                  show the current tags
    /clear                                 drop all tags (unconditional)
    /temp 0.8    /top_p 0.95   /tokens 400 sampling controls
    /help  /quit
    anything else                          an opening line for the model to continue

One-shot instead of a session:

    python3 scripts/inference/story.py CKPT --tags "Genre: Tragedy, Setting: Tavern" \\
        --prompt "The wooden door slammed" --once
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
from brittain.tags import TAG_ORDER, TAG_VALUES, parse_request, render
from brittain.tokenizer_story import StoryTokenizer

DIM, BOLD, RESET = (
    ("\033[2m", "\033[1m", "\033[0m") if sys.stdout.isatty() else ("", "", "")
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--tags", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--once", action="store_true", help="generate once and exit")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-trim", dest="trim", action="store_false",
                        help="stream live instead of trimming to the last sentence")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def build_prompt(tokenizer, tags: dict[str, str], opening: str) -> list[int]:
    """Frame a request the way training framed every document."""
    ids = [tokenizer.special_ids["<|story_start|>"]]
    if tags:
        ids += [
            tokenizer.tags_start,
            *tokenizer.encode(render(tags)),
            tokenizer.tags_end,
        ]
    if opening:
        ids += tokenizer.encode(opening)
    return ids


_SENTENCE_END = tuple('.!?"”’')


def trim_to_sentence(text: str) -> str:
    """Cut back to the last completed sentence.

    Only used when the model ran out of budget without emitting a story
    boundary. It makes the output stop cleanly rather than mid-word; it does not
    give the story an ending, and nothing here can. Every training example is a
    window cut from the middle of a book, so the model has seen almost no real
    endings and has no notion of concluding. That needs complete stories in the
    corpus, not a smarter sampler.
    """
    cut = max((text.rfind(mark) for mark in _SENTENCE_END), default=-1)
    return text[: cut + 1] if cut > 0 else text


def stream(model, tokenizer, device, ids, args) -> str:
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    stop = {tokenizer.eot, tokenizer.story_end, tokenizer.pad}
    utf8 = codecs.getincrementaldecoder("utf-8")("replace")
    emitted = ""
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
            piece = utf8.decode(tokenizer.token_bytes(value))
            emitted += piece
            if not args.trim:
                print(piece, end="", flush=True)
    if args.trim:
        if not closed:
            emitted = trim_to_sentence(emitted)
        print(emitted, end="")
    print()
    if not closed:
        print(f"{DIM}[hit the {args.max_tokens}-token budget without a story "
              f"boundary{'; trimmed to the last sentence' if args.trim else ''}]{RESET}")
    if not emitted.strip():
        print(f"{DIM}[no output — the model emitted a story boundary immediately]{RESET}")
    return emitted


def show_tags(tags: dict[str, str]) -> None:
    if tags:
        print(f"{DIM}tags: {render(tags)}{RESET}")
    else:
        print(f"{DIM}tags: (none — unconditional){RESET}")


def help_text() -> str:
    lines = ["  /tags NAME: VALUE, ...   set tags", "  /tags                    show tags",
             "  /clear                   drop all tags",
             "  /temp X  /top_p X  /tokens N", "  /help  /quit", "", "  known tags:"]
    for name in TAG_ORDER:
        lines.append(f"    {name:9} {', '.join(TAG_VALUES[name])}")
    return "\n".join(lines)


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
    state = checkpoint.get("training_state", {})
    print(f"{BOLD}{path.name}{RESET}  {model.num_params():,} params  "
          f"vocab {tokenizer.vocab_size}  {device.type}")
    if state:
        print(f"{DIM}update {state.get('global_update')}  "
              f"validation {state.get('best_validation', float('nan')):.4f}{RESET}")

    try:
        tags = parse_request(args.tags) if args.tags else {}
    except ValueError as exc:
        raise SystemExit(f"bad --tags: {exc}")

    if args.once:
        show_tags(tags)
        stream(model, tokenizer, device, build_prompt(tokenizer, tags, args.prompt), args)
        return

    print(help_text())
    show_tags(tags)
    while True:
        try:
            line = input(f"\n{BOLD}> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            # Empty line: generate from the tags alone, which is the pure test of
            # whether the conditioning works without a prose hint to lean on.
            stream(model, tokenizer, device, build_prompt(tokenizer, tags, ""), args)
            continue
        if line in ("/quit", "/exit", "/q"):
            return
        if line == "/help":
            print(help_text())
            continue
        if line == "/clear":
            tags = {}
            show_tags(tags)
            continue
        if line.startswith("/tags"):
            rest = line[len("/tags"):].strip()
            if not rest:
                show_tags(tags)
                continue
            try:
                tags = parse_request(rest)
            except ValueError as exc:
                print(f"{DIM}{exc}{RESET}")
            show_tags(tags)
            continue
        for command, attribute, cast in (
            ("/temp", "temperature", float),
            ("/top_p", "top_p", float),
            ("/tokens", "max_tokens", int),
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
            print(f"{DIM}{line}{RESET}", end="")
            stream(model, tokenizer, device, build_prompt(tokenizer, tags, line), args)


if __name__ == "__main__":
    main()
