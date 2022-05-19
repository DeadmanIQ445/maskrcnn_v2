import shutil
from pathlib import Path


def clean_folder_content(folder: Path):
    """Remove content of folder """
    if folder.is_dir():
        shutil.rmtree(folder)

    folder.mkdir()


def get_num_train_image(path_dataset: str):
    path_dataset = Path(path_dataset)

    train_images = path_dataset / "train" / "uchastok_2021"

    return len(list(path for path in train_images.glob("**/*") if path.is_file()))
