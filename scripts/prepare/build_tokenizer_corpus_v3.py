"""Build the filtered Brittain3 tokenizer corpus.

Remote reads are disabled unless ``--allow-remote`` is present. The default
Stack source streams file contents directly through Hugging Face.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.corpus_v3 import (
    CorpusBuilder,
    CorpusRecord,
    classify_path,
    iter_local_repository,
    normalize_language,
    stack_v1_provenance,
    tool_records,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/data/brittain3_tokenizer_corpus.json"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--allow-remote", action="store_true",
        help="permit configured Hugging Face reads",
    )
    parser.add_argument(
        "--remote-source", action="append", default=[],
        help="remote source name to use; repeat it; default is every enabled source",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="multiply all byte quotas; use a small value for a dry run",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the configured seed")
    parser.add_argument(
        "--exclude-corpus", action="append", default=[],
        help="JSONL corpus whose exact and normalized documents must be excluded",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-shortfall", action="store_true",
        help="return success when selected sources cannot fill all quotas",
    )
    return parser.parse_args()


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path):
    with project_path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("format") != "brittain3-tokenizer-corpus-v1":
        raise SystemExit("unsupported tokenizer corpus configuration")
    if abs(sum(config["code_language_shares"].values()) - 1.0) > 1e-9:
        raise SystemExit("code_language_shares must add to 1.0")
    if set(config["category_target_bytes"]) != {
        "code", "documentation", "english", "structured", "tool"
    }:
        raise SystemExit("category_target_bytes must define all five corpus categories")
    for source in config.get("remote_sources", []):
        if source.get("type") != "stack_v1_huggingface":
            continue
        shares = source.get("stream_shares", {})
        buckets = {}
        for language in source["languages"]:
            category, normalized = stack_stream_bucket(language)
            bucket = (category, normalized if category == "code" else category)
            buckets.setdefault(bucket, []).append(language)
        for bucket, languages in buckets.items():
            if len(languages) < 2:
                continue
            total = sum(float(shares.get(language, 1.0)) for language in languages)
            if abs(total - 1.0) > 1e-9:
                raise SystemExit(
                    f"stream_shares for {bucket} must add to 1.0: {languages}"
                )
    return config


def stack_stream_bucket(dataset_language):
    if dataset_language == "markdown":
        return "documentation", "english"
    if dataset_language in {"json", "yaml", "toml", "html", "css", "sql"}:
        return "structured", "other"
    return "code", normalize_language(dataset_language)


def load_exclusions(builder, paths):
    for value in paths:
        path = project_path(value)
        if not path.is_file():
            raise SystemExit(f"exclusion corpus does not exist: {path}")
        print(f"loading exclusions from {path}", flush=True)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    text = json.loads(line)["text"]
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise SystemExit(f"invalid exclusion row at {path}:{line_number}") from exc
                if isinstance(text, str):
                    builder.exclude_text(text)


def add_local_sources(builder, config):
    for source in config.get("local_sources", []):
        if not source.get("enabled", True):
            continue
        root = project_path(source["path"])
        print(f"local source {source['name']}: {root}", flush=True)
        builder.record_source(source["name"], {
            "type": "local_repository", "path": str(root), "license": source["license"]
        })
        try:
            records = iter_local_repository(
                root, source=source["name"], license_name=source["license"],
                seed=config["seed"],
            )
            for record in records:
                builder.add(record)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc


def add_tool_sources(builder, config):
    paths = [project_path(value) for value in config.get("tool_files", [])]
    existing = [path for path in paths if path.is_file()]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(f"tool files not found and skipped: {missing}", flush=True)
    builder.record_source("brittain_tools", {
        "type": "owned_tool_sources", "paths": [str(path) for path in existing],
        "license": "OWNED",
    })
    for record in tool_records(existing, config["seed"]):
        if builder.category_remaining("tool") == 0:
            break
        builder.add(record)


def add_normalized_sources(builder, config):
    required = {"text", "source", "category", "language", "repository", "path", "license"}
    for source in config.get("normalized_sources", []):
        if not source.get("enabled", True):
            continue
        path = project_path(source["path"])
        if not path.is_file():
            raise SystemExit(f"normalized source does not exist: {path}")
        builder.record_source(source["name"], {
            "type": "normalized_jsonl", "path": str(path),
        })
        print(f"normalized source {source['name']}: {path}", flush=True)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"invalid JSON at {path}:{line_number}") from exc
                missing = required - set(row)
                if missing:
                    raise SystemExit(
                        f"normalized row at {path}:{line_number} is missing {sorted(missing)}"
                    )
                builder.add(CorpusRecord(**{name: row[name] for name in required}))


def datasets_module():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("remote sources require: pip install -e '.[train]'") from exc
    return load_dataset


def resolved_huggingface_revision(source):
    try:
        from huggingface_hub import HfApi
        info = HfApi().dataset_info(
            source["dataset"], revision=source.get("revision", "main")
        )
    except Exception as exc:
        raise SystemExit(
            f"could not resolve the dataset revision for {source['dataset']}: {exc}"
        ) from exc
    return info.sha


def add_huggingface_text(builder, source, seed):
    load_dataset = datasets_module()
    revision = resolved_huggingface_revision(source)
    builder.record_source(source["name"], {
        "type": source["type"], "dataset": source["dataset"], "revision": revision,
        "license": source["license"],
    })
    print(f"remote source {source['name']}: streaming {source['dataset']}", flush=True)
    dataset = load_dataset(
        source["dataset"], name=source.get("subset"), split=source.get("split", "train"),
        streaming=True, revision=revision,
    )
    buffer = int(source.get("shuffle_buffer", 0))
    if buffer:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer)
    field = source.get("text_field", "text")
    for index, row in enumerate(dataset):
        if builder.category_remaining(source["category"], source.get("language", "other")) == 0:
            break
        text = row.get(field)
        if not isinstance(text, str):
            continue
        repository = str(row.get("id") or row.get("url") or f"{source['name']}-{index}")
        builder.add(CorpusRecord(
            text=text, source=source["name"], category=source["category"],
            language=source.get("language", "other"), repository=repository,
            path=f"documents/{index}.txt", license=source["license"],
        ))


def add_stack_v1(builder, source, config):
    """Stream the Hugging Face-hosted Stack v1 file contents."""
    load_dataset = datasets_module()
    revision = resolved_huggingface_revision(source)
    source_metadata = {
        "type": source["type"], "dataset": source["dataset"], "revision": revision,
        "license_policy": "all row license values must be in allowed_licenses",
        "directory_aliases": source.get("directory_aliases", {}),
        "stream_shares": source.get("stream_shares", {}),
        "stream_target_bytes": {},
        "stream_accepted_bytes": {},
    }
    builder.record_source(source["name"], source_metadata)
    allowed = set(config["allowed_licenses"])
    bucket_remaining = {}
    for dataset_language in source["languages"]:
        category, language = stack_stream_bucket(dataset_language)
        bucket = (category, language if category == "code" else category)
        if bucket not in bucket_remaining:
            bucket_remaining[bucket] = builder.category_remaining(category, language)
        share = float(source.get("stream_shares", {}).get(dataset_language, 1.0))
        available = bucket_remaining[bucket]
        source_metadata["stream_target_bytes"][dataset_language] = (
            max(1, int(available * share)) if available else 0
        )
    for language_index, dataset_language in enumerate(source["languages"]):
        source_metadata["stream_accepted_bytes"][dataset_language] = 0
        data_directory = source.get("directory_aliases", {}).get(
            dataset_language, dataset_language
        )
        expected_category, expected_language = stack_stream_bucket(dataset_language)
        normalized = normalize_language(dataset_language)
        if builder.category_remaining(expected_category, expected_language) == 0:
            continue
        stream_target = source_metadata["stream_target_bytes"][dataset_language]
        stream_accepted = 0
        print(f"remote source {source['name']}: {dataset_language}", flush=True)
        dataset = load_dataset(
            source["dataset"], data_dir=f"data/{data_directory}",
            split=source.get("split", "train"), streaming=True, revision=revision,
        )
        buffer = int(source.get("shuffle_buffer", 0))
        if buffer:
            dataset = dataset.shuffle(
                seed=config["seed"] + language_index, buffer_size=buffer
            )
        for row in dataset:
            if (
                stream_accepted >= stream_target
                or builder.category_remaining(expected_category, expected_language) == 0
            ):
                break
            provenance = stack_v1_provenance(row, allowed)
            if provenance is None:
                continue
            license_name, repository, path = provenance
            text = row.get("content")
            if not isinstance(text, str):
                continue
            category, path_language = classify_path(path)
            if category is None:
                category, path_language = "code", normalized
            language = normalized if category == "code" else path_language
            if builder.category_remaining(category, language) == 0:
                continue
            accepted = builder.add(CorpusRecord(
                text=text, source=source["name"], category=category,
                language=language, repository=repository, path=path,
                license=license_name,
            ))
            accepted_in_stream_bucket = (
                category == expected_category
                and (
                    category != "code"
                    or normalize_language(language) == expected_language
                )
            )
            if accepted and accepted_in_stream_bucket:
                stream_accepted += len(text.encode("utf-8"))
        source_metadata["stream_accepted_bytes"][dataset_language] = stream_accepted


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    output = project_path(args.output or config["output"])
    report_path = project_path(args.report or config["report"])
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists; use --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    skipped = []
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            builder = CorpusBuilder(config, handle, scale=args.scale)
            load_exclusions(builder, args.exclude_corpus)
            add_tool_sources(builder, config)
            add_local_sources(builder, config)
            add_normalized_sources(builder, config)
            selected = set(args.remote_source)
            for source in config.get("remote_sources", []):
                if not source.get("enabled", True):
                    continue
                if selected and source["name"] not in selected:
                    skipped.append({"name": source["name"], "reason": "not selected"})
                    continue
                if not args.allow_remote:
                    skipped.append({"name": source["name"], "reason": "remote access disabled"})
                    continue
                if source["type"] == "huggingface_text":
                    add_huggingface_text(builder, source, config["seed"])
                elif source["type"] == "stack_v1_huggingface":
                    add_stack_v1(builder, source, config)
                else:
                    raise SystemExit(f"unknown remote source type: {source['type']}")
            report = builder.report(skipped)
        temporary.replace(output)
    except BaseException:
        if temporary.exists():
            print(f"partial corpus kept for inspection: {temporary}", file=sys.stderr)
        raise
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {output} and {report_path}", flush=True)
    if not report["complete"] and not args.allow_shortfall:
        raise SystemExit("corpus quotas are incomplete; inspect the report or use --allow-shortfall")


if __name__ == "__main__":
    main()
