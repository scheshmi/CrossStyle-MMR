import os
from dataclasses import dataclass
from typing import Callable, Optional

from datasets import load_dataset, load_from_disk

from utils.image import preprocess_image, filter_image


@dataclass(frozen=True)
class TaskConfig:
    name: str
    dataset_id: str
    hf_config: Optional[str]
    text_field: str
    raw_label_field: str
    pos_keyword: str
    neg_keyword: str
    label_fn: Callable


def _humor_label(v):
    return 0 if v == "not_funny" else 1


def _offensive_label(v):
    return 0 if v == "not_offensive" else 1


def _identity_label(v):
    return int(v)


TASK_CONFIGS: dict = {
    "sarcasm": TaskConfig(
        name="sarcasm",
        dataset_id="coderchen01/MMSD2.0",
        hf_config="mmsd-clean",
        text_field="text",
        raw_label_field="label",
        pos_keyword="sarcastic",
        neg_keyword="not sarcastic",
        label_fn=_identity_label,
    ),
    "humor": TaskConfig(
        name="humor",
        dataset_id="Ahren09/MMSoc_Memotion",
        hf_config=None,
        text_field="text_corrected",
        raw_label_field="humor",
        pos_keyword="humorous",
        neg_keyword="not humorous",
        label_fn=_humor_label,
    ),
    "metaphor": TaskConfig(
        name="metaphor",
        dataset_id="disk:final_multimet_dataset",
        hf_config=None,
        text_field="text",
        raw_label_field="metaphor",
        pos_keyword="metaphorical",
        neg_keyword="not metaphorical",
        label_fn=_identity_label,
    ),
    "offensive": TaskConfig(
        name="offensive",
        dataset_id="Ahren09/MMSoc_Memotion",
        hf_config=None,
        text_field="text_corrected",
        raw_label_field="offensive",
        pos_keyword="offensive",
        neg_keyword="not offensive",
        label_fn=_offensive_label,
    ),
}


def load_dataset_for_task(
    task: str,
    split: str = "test",
    multimet_disk_path: str = "final_multimet_dataset",
    apply_image_filter: bool = True,
):
    cfg = TASK_CONFIGS[task]

    if cfg.dataset_id.startswith("disk:"):
        ds = load_from_disk(multimet_disk_path)[split]
    else:
        kwargs = {"split": split}
        if cfg.hf_config:
            kwargs["name"] = cfg.hf_config
        ds = load_dataset(cfg.dataset_id, **kwargs)

    if apply_image_filter:
        ds = ds.filter(filter_image)

    ds = ds.map(lambda ex: {"image": preprocess_image(ex["image"])})
    ds = ds.map(lambda ex: {
        "text": ex[cfg.text_field],
        "label": cfg.label_fn(ex[cfg.raw_label_field]),
    })

    return ds
