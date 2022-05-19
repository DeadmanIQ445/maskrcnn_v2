from pathlib import Path

from detectron2.data.datasets import register_coco_instances

is_initialised = False


def register_dataset(dataset_path: Path):
    global is_initialised

    if not is_initialised:
        for name in ['train', 'eval', 'test']:
            register_coco_instances(f"uchastok_{name}", {},
                                    str(dataset_path / name / f"uchastok_{name}2021.json"),
                                    str(dataset_path / name / "uchastok_2021"))
        is_initialised = True
