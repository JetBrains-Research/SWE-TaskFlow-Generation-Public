import hashlib
import json
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from interactive_benchmark_generation.llm import IneligibleProblemError, SplitResult

Splitter = Callable[[str], SplitResult]
ProgressCallback = Callable[[int, int, dict], None]


def normalize_task_text(value: object) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(decoded, str):
            return decoded.strip()
    return text


def build_task_description(record: dict) -> str:
    sections = [normalize_task_text(record["problem_statement"])]
    requirements = record.get("requirements")
    if isinstance(requirements, str) and requirements.strip():
        sections.extend(["## Requirements", normalize_task_text(requirements)])
    interface = record.get("interface")
    if isinstance(interface, str) and interface.strip():
        sections.extend(["## Interface", normalize_task_text(interface)])
    return "\n\n".join(sections)


def load_source_dataset(dataset_name: str, split: str) -> Dataset:
    dataset_path = Path(dataset_name).expanduser()
    if dataset_path.exists():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        if isinstance(loaded, Dataset):
            return loaded
        raise TypeError(f"Unsupported local dataset type: {type(loaded)!r}")

    loaded = load_dataset(dataset_name, split=split)
    if not isinstance(loaded, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(loaded)!r}")
    return loaded


def select_limit(dataset: Dataset, limit: int | None, shuffle_seed: int | None = None) -> Dataset:
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def generate_record(index: int, instance: dict, splitter: Splitter) -> dict:
    record = dict(instance)
    problem_statement = build_task_description(record)
    record["problem_statement"] = problem_statement
    try:
        split_result = splitter(problem_statement)
    except IneligibleProblemError as error:
        debug = {
            "index": index,
            "instance_id": record.get("instance_id"),
            "status": "filtered",
            "filter_category": error.result.category,
            "incremental_units": list(error.result.units),
            "filter_reason": error.result.reason,
        }
        return {"index": index, "status": "filtered", "record": None, "debug": debug}
    except ValueError as error:
        debug = {
            "index": index,
            "instance_id": record.get("instance_id"),
            "status": "rejected",
            "validation_issues": [str(error)],
        }
        return {"index": index, "status": "rejected", "record": None, "debug": debug}
    record["problem_statements"] = split_result.parts
    debug = {
        "index": index,
        "instance_id": record.get("instance_id"),
        "status": "accepted",
        "n_parts": len(split_result.parts),
        "model": split_result.model,
        "filter_category": split_result.filter_category,
        "incremental_units": list(split_result.incremental_units),
        "filter_reason": split_result.filter_reason,
        "rationale": split_result.rationale,
        "coverage_check": split_result.coverage_check,
        "validation_issues": split_result.validation_issues,
    }
    return {"index": index, "status": "accepted", "record": record, "debug": debug}


def build_checkpoint_manifest(dataset: Dataset, config: dict) -> dict:
    sources = []
    for index, instance in enumerate(dataset):
        record = dict(instance)
        description = build_task_description(record)
        sources.append(
            {
                "index": index,
                "instance_id": record.get("instance_id"),
                "task_sha256": hashlib.sha256(description.encode()).hexdigest(),
            }
        )
    payload = {"version": 1, "config": config, "sources": sources}
    fingerprint_input = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return {
        **payload,
        "fingerprint": hashlib.sha256(fingerprint_input.encode()).hexdigest(),
    }


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary_path.replace(path)


def prepare_checkpoint(output_dir: Path, manifest: dict, resume: bool) -> Path:
    checkpoint_dir = output_dir / "checkpoint"
    manifest_path = checkpoint_dir / "manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if not resume:
            raise ValueError(f"Checkpoint already exists in {checkpoint_dir}; pass --resume to continue")
        if existing_manifest.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError("Checkpoint does not match the selected dataset or generation options")
    elif resume:
        raise ValueError(f"No checkpoint found in {checkpoint_dir}")
    else:
        write_json_atomic(manifest_path, manifest)
    results_dir = checkpoint_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def load_checkpoint_results(results_dir: Path, total: int) -> dict[int, dict]:
    results = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            result = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Invalid checkpoint result {path}: {error}") from error
        index = result.get("index")
        status = result.get("status")
        if not isinstance(index, int) or not 0 <= index < total:
            raise ValueError(f"Invalid checkpoint index in {path}")
        if path.name != f"{index:08d}.json" or index in results:
            raise ValueError(f"Invalid or duplicate checkpoint result {path}")
        if status not in {"accepted", "filtered", "rejected"}:
            raise ValueError(f"Invalid checkpoint status in {path}")
        if not isinstance(result.get("debug"), dict):
            raise ValueError(f"Missing checkpoint debug data in {path}")
        if status == "accepted" and not isinstance(result.get("record"), dict):
            raise ValueError(f"Missing accepted record in {path}")
        if status != "accepted" and result.get("record") is not None:
            raise ValueError(f"Unexpected record for {status} result in {path}")
        results[index] = result
    return results


def generate_records(
    dataset: Dataset,
    splitter: Splitter,
    progress_callback: ProgressCallback | None = None,
    workers: int = 1,
    output_dir: Path | None = None,
    resume: bool = False,
    checkpoint_config: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    instances = [dict(instance) for instance in dataset]
    results = {}
    results_dir = None
    if output_dir is not None:
        manifest = build_checkpoint_manifest(dataset, checkpoint_config or {})
        results_dir = prepare_checkpoint(output_dir, manifest, resume)
        results = load_checkpoint_results(results_dir, len(instances))
    pending = [(index, instance) for index, instance in enumerate(instances) if index not in results]
    completed = len(results)
    if progress_callback and completed:
        progress_callback(completed, len(instances), {"resumed": True})
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {executor.submit(generate_record, index, instance, splitter): index for index, instance in pending}
        for future in as_completed(futures):
            result = future.result()
            index = result["index"]
            if results_dir is not None:
                write_json_atomic(results_dir / f"{index:08d}.json", result)
            results[index] = result
            completed += 1
            if progress_callback:
                progress_callback(completed, len(instances), result["debug"])
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    ordered_results = [results[index] for index in range(len(instances))]
    records = [result["record"] for result in ordered_results if result["status"] == "accepted"]
    debug = [result["debug"] for result in ordered_results]
    return records, debug


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, records: list[dict]) -> None:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n")


def quote_markdown(text: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def write_prompts_markdown(path: Path, records: list[dict], metadata: dict) -> None:
    source_dataset = metadata.get("source_dataset", "unknown")
    split = metadata.get("split", "unknown")
    shuffle_seed = metadata.get("shuffle_seed")
    seed_suffix = f", shuffle seed `{shuffle_seed}`" if shuffle_seed is not None else ""
    sections = [
        "# Single-prompt and multi-prompt inputs",
        "",
        f"Dataset: `{source_dataset}`, split `{split}`{seed_suffix}.",
    ]
    for index, record in enumerate(records, start=1):
        sections.extend(
            [
                "",
                f"## {index}. `{record.get('instance_id', index - 1)}`",
                "",
                "### Single prompt",
                "",
                quote_markdown(str(record["problem_statement"])),
                "",
                "### Multi prompt",
            ]
        )
        for part_index, part in enumerate(record["problem_statements"], start=1):
            sections.extend(
                [
                    "",
                    f"#### Prompt {part_index}",
                    "",
                    quote_markdown(str(part)),
                ]
            )
    path.write_text("\n".join(sections) + "\n")


def save_generated_dataset(
    records: list[dict],
    debug: list[dict],
    output_dir: Path,
    metadata: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    Dataset.from_list(records).save_to_disk(str(dataset_dir))
    write_jsonl(output_dir / "preview.jsonl", records)
    write_json(output_dir / "preview.json", records)
    write_json(output_dir / "debug.json", debug)
    write_prompts_markdown(output_dir / "prompts.md", records, metadata)
    filtered_instances = sum(item.get("status") == "filtered" for item in debug)
    rejected_instances = sum(item.get("status") == "rejected" for item in debug)
    write_json(
        output_dir / "metadata.json",
        {
            **metadata,
            "created_at": int(time.time()),
            "n_instances": len(records),
            "n_filtered_instances": filtered_instances,
            "n_rejected_instances": rejected_instances,
            "dataset_dir": str(dataset_dir),
        },
    )
    return dataset_dir
