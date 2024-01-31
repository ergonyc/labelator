import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def dump_label_encoder(le: LabelEncoder, path: Path):
    if not path.is_file():
        path = path / "label_encoder.pkl"

    if not path.parent.exists():
        path.parent.mkdir(parents=True)

    with path.open("wb") as f:
        pickle.dump(le, f)


def load_label_encoder(path: Path) -> LabelEncoder:
    if not path.is_file():
        path = path / "label_encoder.pkl"

    with path.open("rb") as f:
        return pickle.load(f)
