import os
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console

from interactive_benchmark_generation.dataset_io import (
    generate_records,
    load_source_dataset,
    save_generated_dataset,
    select_limit,
)
from interactive_benchmark_generation.llm import split_problem_statement
from interactive_benchmark_generation.presets import PRESETS, resolve_dataset
from interactive_benchmark_generation.qa_evaluation import (
    evaluate_agent_results,
    evaluate_agent_run,
)
from interactive_benchmark_generation.qa_io import (
    load_generation_bundle,
    validate_answer_contracts,
    validate_golden_proofs,
    validate_question_complexity,
    write_qa_bundle,
)
from interactive_benchmark_generation.qa_judge import (
    judge_bundle_answers,
    judge_bundle_results,
)
from interactive_benchmark_generation.qa_multistep_builder import (
    build_hybrid_dataset,
)
from interactive_benchmark_generation.qa_replacement import (
    apply_question_replacements,
    build_question_replacement_request,
)
from interactive_benchmark_generation.qa_validation import (
    import_adversarial_review,
    import_answer_judge_report,
    import_golden_proof_execution_report,
    import_judge_report,
    import_probe_validation_report,
    import_proof_execution_report,
    validate_qa_bundle,
)

app = typer.Typer(add_completion=False)
console = Console()


def configure_environment(env_file: Path | None, litellm_base_url: str | None) -> None:
    if env_file and env_file.exists():
        load_dotenv(env_file, override=True)

    litellm_base_url = litellm_base_url or os.getenv("LITELLM_BASE_URL")
    if litellm_base_url:
        os.environ["LITELLM_BASE_URL"] = litellm_base_url
        os.environ["OPENAI_BASE_URL"] = litellm_base_url
        os.environ["OPENAI_API_BASE"] = litellm_base_url

    if os.getenv("LITELLM_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["LITELLM_API_KEY"]


@app.command()
def generate(
    output_dir: Annotated[Path, typer.Option(help="Directory where the generated dataset will be saved.")],
    preset: Annotated[
        str | None,
        typer.Option(help=f"Dataset preset: {', '.join(sorted(PRESETS))}."),
    ] = "swe-bench-verified",
    dataset_name: Annotated[str | None, typer.Option(help="HF dataset name or local load_from_disk path.")] = None,
    split: Annotated[str | None, typer.Option(help="Dataset split.")] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Maximum number of instances to transform.")] = None,
    shuffle_seed: Annotated[
        int | None,
        typer.Option(help="Shuffle the source dataset with this seed before applying --limit."),
    ] = None,
    n_parts: Annotated[int, typer.Option(min=2, help="Number of interactive prompts per instance.")] = 3,
    model: Annotated[
        str, typer.Option(help="LiteLLM model name for generation and validation.")
    ] = "openai/gpt-5.6-luna",
    fallback_model: Annotated[
        list[str] | None,
        typer.Option(help="Fallback LiteLLM model name. Can be passed multiple times."),
    ] = None,
    max_attempts: Annotated[int, typer.Option(min=1, help="Maximum generation retries per instance.")] = 3,
    request_timeout: Annotated[
        float | None,
        typer.Option(min=1.0, help="Per-request LLM timeout in seconds."),
    ] = 90.0,
    retry_delay: Annotated[
        float,
        typer.Option(min=0.0, help="Initial retry delay for model call failures, with exponential backoff."),
    ] = 5.0,
    workers: Annotated[int, typer.Option(min=1, help="Number of concurrent generation workers.")] = 8,
    resume: Annotated[bool, typer.Option(help="Resume from checkpoints in the output directory.")] = False,
    env_file: Annotated[Path | None, typer.Option(help="Optional .env file.")] = Path(".env"),
    litellm_base_url: Annotated[str | None, typer.Option(help="LiteLLM base URL.")] = None,
    llm_validation: Annotated[bool, typer.Option(help="Run a second LLM validation pass.")] = True,
) -> None:
    configure_environment(env_file, litellm_base_url)
    resolved_dataset_name, resolved_split = resolve_dataset(preset, dataset_name, split)
    dataset = select_limit(load_source_dataset(resolved_dataset_name, resolved_split), limit, shuffle_seed)

    def splitter(problem_statement: str):
        return split_problem_statement(
            problem_statement=problem_statement,
            n_parts=n_parts,
            model=model,
            max_attempts=max_attempts,
            llm_validation=llm_validation,
            fallback_models=fallback_model or [],
            request_timeout=request_timeout,
            retry_delay=retry_delay,
        )

    checkpoint_config = {
        "source_dataset": resolved_dataset_name,
        "split": resolved_split,
        "preset": preset,
        "limit": limit,
        "model": model,
        "fallback_models": fallback_model or [],
        "n_parts": n_parts,
        "shuffle_seed": shuffle_seed,
        "llm_validation": llm_validation,
        "max_attempts": max_attempts,
        "request_timeout": request_timeout,
        "retry_delay": retry_delay,
    }

    def show_progress(completed: int, total: int, result: dict) -> None:
        if result.get("resumed"):
            console.print(f"Resumed {completed}/{total} completed splits")
            return
        instance_id = result.get("instance_id", result.get("index", completed - 1))
        console.print(f"Completed split {completed}/{total}: {instance_id} ({result['status']})")

    records, debug = generate_records(
        dataset,
        splitter,
        show_progress,
        workers=workers,
        output_dir=output_dir,
        resume=resume,
        checkpoint_config=checkpoint_config,
    )
    dataset_dir = save_generated_dataset(
        records=records,
        debug=debug,
        output_dir=output_dir,
        metadata={
            **checkpoint_config,
            "workers": workers,
            "resumed": resume,
        },
    )
    console.print(f"Saved generated dataset to {dataset_dir}")


@app.command()
def inspect(
    dataset_dir: Annotated[Path, typer.Option(help="Generated dataset directory saved by save_to_disk.")],
    index: Annotated[int, typer.Option(min=0, help="Instance index to inspect.")] = 0,
) -> None:
    dataset = load_source_dataset(str(dataset_dir), split="train")
    instance = dict(dataset[index])
    console.rule(str(instance.get("instance_id", index)))
    console.print("[bold]Original problem_statement[/bold]")
    console.print(instance.get("problem_statement", ""))
    console.print()
    console.print("[bold]Generated problem_statements[/bold]")
    for part_index, part in enumerate(instance.get("problem_statements", []), start=1):
        console.print(f"\n[bold]{part_index}.[/bold] {part}")


@app.command("qa-ingest")
def qa_ingest(
    bundle: Annotated[Path, typer.Option(help="Structured QA generation bundle JSON file or directory.")],
    output_dir: Annotated[Path, typer.Option(help="Directory where QA artifacts will be written.")],
    trajectory: Annotated[
        Path | None,
        typer.Option(help="Optional mini-SWE-agent trajectory JSON used to record provenance."),
    ] = None,
) -> None:
    generation_bundle = load_generation_bundle(bundle, prune=True)
    validate_answer_contracts(generation_bundle)
    validate_question_complexity(generation_bundle)
    write_qa_bundle(generation_bundle, output_dir, trajectory)
    dropped = (generation_bundle.debug or {}).get("ingestion_dropped_questions") or []
    for item in dropped:
        console.print(f"Dropped {item['question_id']}: {item['reason']} — {item['detail']}")
    console.print(
        f"Wrote QA bundle to {output_dir} "
        f"({len(generation_bundle.questions)} questions, {len(dropped)} dropped)"
    )


@app.command("build-qa-multistep")
def build_qa_multistep(
    multi_prompt_records: Annotated[
        Path,
        typer.Option(
            help="Multi-step preview JSON or JSONL containing problem_statements."
        ),
    ],
    qa_run_root: Annotated[
        Path,
        typer.Option(help="Validated QA bundle or QA controller generation root."),
    ],
    task_root: Annotated[
        Path,
        typer.Option(help="Original SWE-bench Pro task root."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(help="New or empty output directory for public artifacts."),
    ],
    task_name: Annotated[
        list[str] | None,
        typer.Option(help="Full instance ID to include. May be passed multiple times."),
    ] = None,
    allow_unused_qa: Annotated[
        bool,
        typer.Option(help="Allow QA bundles whose instance is absent from the source."),
    ] = False,
    verify_rebuild: Annotated[
        bool,
        typer.Option(
            help="Compare rebuilt bytes with an existing output without writing."
        ),
    ] = False,
) -> None:
    result = build_hybrid_dataset(
        multi_prompt_records=multi_prompt_records,
        qa_run_root=qa_run_root,
        task_root=task_root,
        output_dir=output_dir,
        task_names=task_name,
        allow_unused_qa=allow_unused_qa,
        verify_rebuild=verify_rebuild,
    )
    action = "Verified" if verify_rebuild else "Wrote"
    coverage = result["coverage"]
    console.print(
        f"{action} {coverage['multi_step_instances']} hybrid records at "
        f"{output_dir}; {coverage['instances_with_qa']} with QA and "
        f"{coverage['instances_without_qa']} without QA"
    )


@app.command("qa-validate")
def qa_validate(
    bundle_dir: Annotated[Path, typer.Option(help="Pre-generated QA bundle directory.")],
    repository: Annotated[Path, typer.Option(help="Repository root where probes are run.")],
    timeout: Annotated[
        float,
        typer.Option(min=0.1, help="Timeout in seconds for each probe invocation."),
    ] = 30.0,
) -> None:
    report = validate_qa_bundle(bundle_dir, repository, timeout)
    console.print(
        f"Validated {report['n_valid']}/{report['n_questions']} questions; "
        f"repository unchanged: {report['repository_state_unchanged']}"
    )
    if not report["valid"]:
        raise typer.Exit(code=1)


@app.command("qa-import-validation")
def qa_import_validation(
    bundle_dir: Annotated[Path, typer.Option(help="Pre-generated QA bundle directory.")],
    report: Annotated[Path, typer.Option(help="Harbor verifier output JSON.")],
) -> None:
    imported = import_probe_validation_report(bundle_dir, report)
    console.print(
        f"Imported validation for {imported['n_questions']} questions; "
        f"valid: {imported['valid']}"
    )
    if not imported["valid"]:
        raise typer.Exit(code=1)


@app.command("qa-import-review")
def qa_import_review(
    bundle_dir: Annotated[Path, typer.Option(help="Pre-generated QA bundle directory.")],
    report: Annotated[Path, typer.Option(help="Adversarial reviewer output JSON.")],
) -> None:
    imported = import_adversarial_review(bundle_dir, report)
    console.print(
        f"Adversarial review accepted {imported['n_accepted']}/"
        f"{imported['n_questions']} questions"
    )


@app.command("qa-validate-golden-proofs")
def qa_validate_golden_proofs(
    bundle_dir: Annotated[
        Path,
        typer.Option(
            help="Proof-enabled QA bundle directory for structural and hash preflight only."
        ),
    ],
) -> None:
    bundle = load_generation_bundle(bundle_dir)
    report = validate_golden_proofs(
        bundle,
        bundle_dir,
        require_executable=False,
    )
    console.print(
        f"Structurally validated {report['n_valid']} golden proofs for "
        f"{report['n_questions']} questions"
    )


@app.command("qa-import-golden-proof-executions")
def qa_import_golden_proof_executions(
    bundle_dir: Annotated[Path, typer.Option(help="Raw v2 QA bundle directory.")],
    report: Annotated[
        Path,
        typer.Option(help="Trusted Harbor golden proof execution report."),
    ],
    drop_invalid: Annotated[
        bool,
        typer.Option(
            "--drop-invalid/--keep-invalid",
            help="Also drop questions whose golden proof failed trusted execution, "
            "so the reviewer never sees a question it could contradict.",
        ),
    ] = False,
    require_patch_stable: Annotated[
        bool,
        typer.Option(
            "--require-patch-stable/--allow-unknown-patch-stability",
            help="When dropping invalid questions, require a trusted "
            "post-patch success verdict.",
        ),
    ] = False,
) -> None:
    imported = import_golden_proof_execution_report(
        bundle_dir,
        report,
        drop_invalid=drop_invalid,
        require_patch_stable=require_patch_stable,
    )
    console.print(
        f"Imported and attached {imported['n_executions']} trusted golden proof "
        f"validations for {imported['n_questions']} questions"
    )
    if imported.get("n_questions_dropped"):
        console.print(
            f"Dropped {imported['n_questions_dropped']} question(s) whose golden proof "
            f"failed trusted execution: {', '.join(imported['dropped_question_ids'])}"
        )


@app.command("qa-build-question-replacement-request")
def qa_build_question_replacement_request(
    bundle_dir: Annotated[
        Path,
        typer.Option(help="Proof-enabled v2 QA bundle directory."),
    ],
    output: Annotated[
        Path,
        typer.Option(help="Path for the strict replacement request JSON."),
    ],
    question_id: Annotated[
        list[str],
        typer.Option(help="Question to repair or regenerate. Repeat as needed."),
    ],
    mode: Annotated[
        str,
        typer.Option(help="Replacement mode: repair or regeneration."),
    ] = "repair",
    attempt: Annotated[
        int,
        typer.Option(min=1, help="Current per-question attempt number."),
    ] = 1,
    max_attempts: Annotated[
        int,
        typer.Option(min=1, help="Maximum per-question attempts."),
    ] = 3,
    diagnostics: Annotated[
        Path | None,
        typer.Option(
            help="Optional sanitized pristine/post-patch diagnostics JSON."
        ),
    ] = None,
) -> None:
    request = build_question_replacement_request(
        bundle_dir,
        output,
        mode=mode,
        question_ids=question_id,
        attempt=attempt,
        max_attempts=max_attempts,
        diagnostics_path=diagnostics,
    )
    console.print(
        f"Wrote {request['protocol']} for "
        f"{len(request['question_ids'])} question(s) to {output}"
    )


@app.command("qa-apply-question-replacements")
def qa_apply_question_replacements(
    bundle_dir: Annotated[
        Path,
        typer.Option(help="Proof-enabled v2 QA bundle directory."),
    ],
    request: Annotated[
        Path,
        typer.Option(help="Strict repair or regeneration request JSON."),
    ],
    replacements: Annotated[
        Path,
        typer.Option(help="Model-produced qa_question_replacements/v1 JSON."),
    ],
) -> None:
    summary = apply_question_replacements(
        bundle_dir,
        request,
        replacements,
    )
    console.print(
        f"Applied {summary['n_applied']} replacement(s); "
        f"retained {summary['n_retained_for_retry']} for retry; "
        f"dropped {summary['n_dropped']}"
    )


@app.command("qa-import-proof-executions")
def qa_import_proof_executions(
    bundle_dir: Annotated[Path, typer.Option(help="Proof-enabled QA bundle directory.")],
    report: Annotated[Path, typer.Option(help="Trusted Harbor proof execution report.")],
    stage: Annotated[
        str,
        typer.Option(help="Import stage: agent or combined."),
    ] = "combined",
) -> None:
    imported = import_proof_execution_report(bundle_dir, report, stage=stage)
    console.print(
        f"Imported {imported['n_executions']} proof executions for "
        f"{imported['n_questions']} questions"
    )


@app.command("qa-import-judge")
def qa_import_judge(
    bundle_dir: Annotated[Path, typer.Option(help="Proof-enabled QA bundle directory.")],
    report: Annotated[Path, typer.Option(help="Sol judge report JSON.")],
) -> None:
    imported = import_judge_report(bundle_dir, report)
    console.print(f"Imported {len(imported['assessments'])} Sol judge assessments")


@app.command("qa-import-answer-judge")
def qa_import_answer_judge(
    bundle_dir: Annotated[
        Path,
        typer.Option(help="Proof-enabled QA bundle directory."),
    ],
    report: Annotated[
        Path,
        typer.Option(help="Sol answer judge report JSON."),
    ],
) -> None:
    imported = import_answer_judge_report(bundle_dir, report)
    console.print(
        f"Imported {len(imported['assessments'])} Sol answer assessments"
    )


@app.command("qa-answer-judge")
def qa_answer_judge(
    bundle_dir: Annotated[
        Path,
        typer.Option(help="Proof-enabled QA bundle directory."),
    ],
    agent_results: Annotated[
        Path,
        typer.Option(help="Agent answers directory, JSON, JSONL, or trajectory."),
    ],
    golden_executions: Annotated[
        Path,
        typer.Option(help="Trusted Harbor golden proof execution report."),
    ],
    output: Annotated[Path, typer.Option(help="Judge report JSON path.")],
    checkpoint_context: Annotated[
        Path | None,
        typer.Option(help="Optional checkpoint and SWE context JSON."),
    ] = None,
    model: Annotated[
        str,
        typer.Option(help="LiteLLM judge model."),
    ] = "openai/gpt-5.6-sol",
    reasoning_effort: Annotated[
        str,
        typer.Option(help="Judge reasoning effort."),
    ] = "medium",
    max_attempts: Annotated[int, typer.Option(min=1)] = 3,
    request_timeout: Annotated[
        float,
        typer.Option(min=1.0),
    ] = 90.0,
    retry_delay: Annotated[float, typer.Option(min=0.0)] = 1.0,
    env_file: Annotated[
        Path | None,
        typer.Option(help="Optional .env file."),
    ] = Path(".env"),
    litellm_base_url: Annotated[
        str | None,
        typer.Option(help="LiteLLM base URL."),
    ] = None,
) -> None:
    configure_environment(env_file, litellm_base_url)
    report = judge_bundle_answers(
        bundle_dir=bundle_dir,
        agent_results_path=agent_results,
        golden_executions_path=golden_executions,
        output_path=output,
        checkpoint_context_path=checkpoint_context,
        model=model,
        reasoning_effort=reasoning_effort,
        max_attempts=max_attempts,
        request_timeout=request_timeout,
        retry_delay=retry_delay,
    )
    console.print(
        f"Judged {len(report['assessments'])} answers with "
        f"{len(report['errors'])} errors"
    )
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command("qa-judge")
def qa_judge(
    bundle_dir: Annotated[Path, typer.Option(help="Proof-enabled QA bundle directory.")],
    agent_results: Annotated[
        Path,
        typer.Option(help="Agent answers/proofs directory, JSON, JSONL, or trajectory."),
    ],
    proof_executions: Annotated[
        Path,
        typer.Option(help="Trusted Harbor proof execution report."),
    ],
    output: Annotated[Path, typer.Option(help="Judge report JSON path.")],
    model: Annotated[str, typer.Option(help="LiteLLM judge model.")] = "openai/gpt-5.6-sol",
    reasoning_effort: Annotated[
        str,
        typer.Option(help="Judge reasoning effort."),
    ] = "medium",
    max_attempts: Annotated[int, typer.Option(min=1)] = 3,
    request_timeout: Annotated[float, typer.Option(min=1.0)] = 90.0,
    retry_delay: Annotated[float, typer.Option(min=0.0)] = 1.0,
    env_file: Annotated[Path | None, typer.Option(help="Optional .env file.")] = Path(".env"),
    litellm_base_url: Annotated[str | None, typer.Option(help="LiteLLM base URL.")] = None,
) -> None:
    configure_environment(env_file, litellm_base_url)
    report = judge_bundle_results(
        bundle_dir=bundle_dir,
        agent_results_path=agent_results,
        proof_executions_path=proof_executions,
        output_path=output,
        model=model,
        reasoning_effort=reasoning_effort,
        max_attempts=max_attempts,
        request_timeout=request_timeout,
        retry_delay=retry_delay,
    )
    console.print(
        f"Judged {len(report['assessments'])} questions with "
        f"{len(report['errors'])} errors"
    )
    if report["errors"]:
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    bundle_dir: Annotated[Path, typer.Option(help="Validated QA bundle directory.")],
    agent_results: Annotated[Path, typer.Option(help="Agent result JSON or JSONL file.")],
    output_dir: Annotated[Path, typer.Option(help="Directory where evaluation reports will be written.")],
    proof_executions: Annotated[
        Path | None,
        typer.Option(help="Imported trusted proof execution report."),
    ] = None,
    judge_report: Annotated[
        Path | None,
        typer.Option(help="Imported Sol judge report."),
    ] = None,
    judge_cost: Annotated[
        float | None,
        typer.Option(min=0, help="Trusted external judge cost override."),
    ] = None,
) -> None:
    report = evaluate_agent_results(
        bundle_dir,
        agent_results,
        output_dir,
        proof_executions,
        judge_report,
        judge_cost,
    )
    if report.get("evaluation_mode") == "proof_judged_v1":
        console.print(
            f"Answer accuracy: {report['n_answer_correct']}/{report['n_scored']} "
            f"({report['answer_accuracy']:.1%}); answer plus proof accuracy: "
            f"{report['n_answer_plus_proof_correct']}/{report['n_scored']} "
            f"({report['answer_plus_proof_accuracy']:.1%}); "
            f"dataset issues: {report['n_dataset_issues']}"
        )
        return
    if report.get("evaluation_mode") == "answer_judged_v1":
        console.print(
            f"Answer accuracy: {report['n_answer_correct']}/"
            f"{report['n_scored']} ({report['answer_accuracy']:.1%}); "
            f"dataset issues: {report['n_dataset_issues']}"
        )
        return
    console.print(
        f"Correct answers: {report['n_correct']}/{report['n_questions']} "
        f"({report['correct_proportion']:.1%}); "
        f"average steps per answer: {report['average_steps_per_answer']:.2f}"
    )


@app.command("evaluate-batch")
def evaluate_batch(
    bundle_root: Annotated[Path, typer.Option(help="QA run directory with instance bundles.")],
    agent_run_dir: Annotated[Path, typer.Option(help="Harbor run containing agent trajectories.")],
    output_dir: Annotated[Path, typer.Option(help="Directory where evaluation reports will be written.")],
) -> None:
    report = evaluate_agent_run(bundle_root, agent_run_dir, output_dir)
    console.print(
        f"Correct answers: {report['n_correct']}/{report['n_questions']} "
        f"({report['correct_proportion']:.1%}) across {report['n_instances']} instances; "
        f"average steps per answer: {report['average_steps_per_answer']:.2f}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
