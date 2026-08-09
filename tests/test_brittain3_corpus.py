import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brittain.corpus_v3 import (
    CorpusBuilder,
    CorpusRecord,
    accepted_license,
    classify_path,
    iter_local_repository,
    tool_records,
    stack_v1_provenance,
)
import scripts.prepare.build_tokenizer_corpus_v3 as corpus_script
from scripts.prepare.build_tokenizer_corpus_v3 import load_config, stack_stream_bucket


def config():
    return {
        "seed": 9,
        "category_target_bytes": {
            "code": 10_000, "documentation": 10_000, "english": 10_000,
            "structured": 10_000, "tool": 10_000,
        },
        "code_language_shares": {
            "python": 0.5, "typescript": 0.1, "javascript": 0.1,
            "c_cpp": 0.05, "rust": 0.05, "go": 0.05,
            "java_kotlin": 0.05, "shell": 0.05, "other": 0.05,
        },
        "allowed_licenses": ["MIT", "OWNED"],
        "filters": {
            "minimum_document_bytes": 20,
            "maximum_document_bytes": 2_000,
            "maximum_line_bytes": 500,
            "minimum_printable_fraction": 0.9,
            "max_repository_bytes": 5_000,
            "excluded_repository_substrings": ["human_eval"],
        },
    }


def record(text, **changes):
    values = {
        "text": text,
        "source": "fixture",
        "category": "code",
        "language": "python",
        "repository": "owner/repo",
        "path": "src/app.py",
        "license": "MIT",
    }
    values.update(changes)
    return CorpusRecord(**values)


def test_path_classification():
    assert classify_path("src/app.py") == ("code", "python")
    assert classify_path("README.md") == ("documentation", "english")
    assert classify_path("pyproject.toml")[0] == "structured"
    assert classify_path("image.png")[0] is None


def test_stack_v1_dedup_provenance_fields():
    allowed = {"MIT", "Apache-2.0"}
    row = {
        "hexsha": "abc",
        "ext": "py",
        "max_stars_repo_licenses": ["mit"],
        "max_stars_repo_name": "owner/project",
        "max_stars_repo_path": "src/app.py",
    }
    assert accepted_license(["apache-2.0"], allowed) == "Apache-2.0"
    assert stack_v1_provenance(row, allowed) == (
        "MIT", "owner/project", "src/app.py"
    )


def test_stack_grouped_streams_have_valid_sub_mixtures():
    assert stack_stream_bucket("c++") == ("code", "c_cpp")
    assert stack_stream_bucket("kotlin") == ("code", "java_kotlin")
    assert stack_stream_bucket("powershell") == ("code", "shell")
    assert stack_stream_bucket("json") == ("structured", "other")
    configured = load_config(PROJECT_ROOT / "configs/data/brittain3_tokenizer_corpus.json")
    stack = next(
        source for source in configured["remote_sources"]
        if source["name"] == "the-stack-dedup"
    )
    assert stack["directory_aliases"]["c++"] == "cpp"
    assert stack["stream_shares"]["c++"] + stack["stream_shares"]["c"] == 1.0


def test_stack_grouped_stream_cap_keeps_later_language(monkeypatch):
    output = io.StringIO()
    builder = CorpusBuilder(config(), output)
    local_code = record(
        "int local_value = 1;\nint local_result = local_value + 1;\n",
        language="c_cpp", path="local.c",
    )
    assert builder.add(local_code)
    remote_capacity = builder.category_remaining("code", "c_cpp")

    def fake_load_dataset(_dataset, *, data_dir, **_kwargs):
        suffix = ".cpp" if data_dir == "data/cpp" else ".c"
        common = {
            "licenses": ["MIT"],
            "repository_name": f"owner/{data_dir}",
        }
        return [
            {
                **common,
                "content": f"# Documentation for {data_dir}\n" + "Useful text.\n" * 20,
                "path": f"docs/{data_dir}-README.md",
            },
            {
                **common,
                "content": f"/* {data_dir} */\n" + "int value = 1;\n" * 20,
                "path": f"src/value{suffix}",
            },
        ]

    monkeypatch.setattr(corpus_script, "datasets_module", lambda: fake_load_dataset)
    monkeypatch.setattr(corpus_script, "resolved_huggingface_revision", lambda _source: "sha")
    source = {
        "name": "stack", "type": "stack_v1_huggingface", "dataset": "fixture",
        "languages": ["c++", "c"], "directory_aliases": {"c++": "cpp"},
        "stream_shares": {"c++": 0.5, "c": 0.5},
    }
    corpus_script.add_stack_v1(builder, source, config())
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    stack_code = [
        row for row in rows
        if row["source"] == "stack" and row["category"] == "code"
    ]
    assert {Path(row["path"]).suffix for row in stack_code} == {".cpp", ".c"}
    stream_bytes = builder.source_metadata["stack"]["stream_accepted_bytes"]
    assert stream_bytes["c++"] > 0
    assert stream_bytes["c"] > 0
    assert sum(stream_bytes.values()) == sum(
        len(row["text"].encode("utf-8")) for row in stack_code
    )
    stream_targets = builder.source_metadata["stack"]["stream_target_bytes"]
    assert stream_targets["c++"] + stream_targets["c"] <= remote_capacity


def test_report_accepts_only_sub_per_mille_rounding_shortfall():
    builder = CorpusBuilder(config(), io.StringIO())
    builder.accepted_bytes.update(builder.category_targets)
    builder.language_bytes.update(builder.language_targets)
    builder.accepted_bytes["code"] -= 5
    builder.language_bytes["python"] -= 5
    report = builder.report()
    assert report["exact_complete"] is False
    assert report["complete"] is True
    assert report["completion_tolerance_fraction"] == 0.001
    builder.language_bytes["python"] -= 1
    assert builder.report()["complete"] is False


def test_builder_filters_license_secret_generated_and_duplicates():
    output = io.StringIO()
    builder = CorpusBuilder(config(), output)
    accepted = "def useful_function(value):\n    return value + 1\n"
    assert builder.add(record(accepted))
    assert not builder.add(record(accepted, path="src/copy.py"))
    assert not builder.add(record(
        "def private():\n    token = 'ghp_abcdefghijklmnopqrstuvwxyz1234567890'\n",
        path="src/private.py",
    ))
    assert not builder.add(record(
        "# This file was generated. Do not edit.\ndef generated():\n    pass\n",
        path="src/generated.py",
    ))
    assert not builder.add(record(
        "def restricted_code():\n    return False\n", license="GPL-3.0"
    ))
    assert not builder.add(record(
        "def benchmark_solution():\n    return 42\n",
        repository="evaluation/human_eval_solutions",
    ))
    row = json.loads(output.getvalue())
    assert row["text"] == accepted
    assert builder.rejected["exact_duplicate"] == 1
    assert builder.rejected["secret"] == 1
    assert builder.rejected["generated"] == 1
    assert builder.rejected["license"] == 1
    assert builder.rejected["evaluation_contamination"] == 1


def test_builder_excludes_training_corpus_from_holdout():
    output = io.StringIO()
    builder = CorpusBuilder(config(), output)
    text = "def training_only(value):\n    return value\n"
    builder.exclude_text(text)
    assert not builder.add(record(text))
    assert builder.rejected["excluded_corpus_duplicate"] == 1


def test_local_repository_scan_is_stable_and_skips_data(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "src" / "app.py").write_text("def app():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("A useful project description.\n", encoding="utf-8")
    (tmp_path / "data" / "leak.py").write_text("SECRET_DATA = True\n", encoding="utf-8")
    first = list(iter_local_repository(
        tmp_path, source="local", license_name="OWNED", seed=4
    ))
    second = list(iter_local_repository(
        tmp_path, source="local", license_name="OWNED", seed=4
    ))
    assert first == second
    assert {item.path for item in first} == {"src/app.py", "README.md"}


def test_tool_examples_use_atomic_protocol_text(tmp_path):
    source = tmp_path / "tools.js"
    source.write_text(
        "const tools = [{ type: 'function', function: { name: 'read_file' } }];\n",
        encoding="utf-8",
    )
    records = list(tool_records([source], seed=2))
    synthetic = next(item for item in records if item.path.startswith("synthetic/"))
    assert "<|tool_call|>" in synthetic.text
    assert "<|tool_result|>" in synthetic.text
    payload = synthetic.text.split("<|tool_call|>", 1)[1].split("<|end_message|>", 1)[0]
    assert json.loads(payload)["name"] == "read_file"
