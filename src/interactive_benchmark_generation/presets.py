from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetPreset:
    dataset_name: str
    split: str


PRESETS = {
    "swe-bench-verified": DatasetPreset(dataset_name="SWE-bench/SWE-bench_Verified", split="test"),
    "rebench-v2": DatasetPreset(dataset_name="nebius/SWE-rebench", split="test"),
    "swe-bench-pro": DatasetPreset(dataset_name="ScaleAI/SWE-bench_Pro", split="test"),
    "swe-atlas": DatasetPreset(dataset_name="SWE-bench/SWE-Atlas", split="test"),
}


def resolve_dataset(preset: str | None, dataset_name: str | None, split: str | None) -> tuple[str, str]:
    if dataset_name:
        if not split:
            raise ValueError("--split is required when --dataset-name is used")
        return dataset_name, split

    if not preset:
        raise ValueError("Either --preset or --dataset-name must be provided")

    if preset not in PRESETS:
        options = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {preset!r}. Available presets: {options}")

    dataset_preset = PRESETS[preset]
    return dataset_preset.dataset_name, split or dataset_preset.split
