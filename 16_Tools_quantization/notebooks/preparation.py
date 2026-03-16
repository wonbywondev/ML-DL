from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple
import xml.etree.ElementTree as ET

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import v2


SplitName = Literal["trainval", "test"]


@dataclass(frozen=True)
class ProjectPaths:
    root_dir: Path
    data_dir: Path
    raw_dir: Path
    image_dir: Path
    annot_dir: Path
    bbox_dir: Path
    trainval_label_path: Path
    test_label_path: Path
    model_dir: Path
    output_dir: Path


def get_paths(ensure_dirs: bool = False) -> ProjectPaths:
    """
    notebooks/make_dataset.py 기준으로 프로젝트 루트를 계산한다.
    """
    root_dir = Path(__file__).resolve().parents[1]
    data_dir = root_dir / "data"
    raw_dir = data_dir / "raw"
    image_dir = raw_dir / "images" / "images"
    annot_dir = raw_dir / "annotations" / "annotations"
    bbox_dir = annot_dir / "xmls"
    model_dir = root_dir / "models"
    output_dir = root_dir / "outputs"

    if ensure_dirs:
        model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        root_dir=root_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        image_dir=image_dir,
        annot_dir=annot_dir,
        bbox_dir=bbox_dir,
        trainval_label_path=annot_dir / "trainval.txt",
        test_label_path=annot_dir / "test.txt",
        model_dir=model_dir,
        output_dir=output_dir,
    )


def get_model_paths(prefix: str = "model") -> Dict[str, Path]:
    """
    modeling/inference에서 공통으로 쓰는 모델 파일 경로.
    """
    p = get_paths(ensure_dirs=True)

    return {
        "basic_pth": p.model_dir / f"{prefix}.pth",
        "quant_pth": p.model_dir / f"{prefix}_quantized.pth",
        "onnx": p.model_dir / f"{prefix}.onnx",
    }


def _read_label_dict(label_txt_path: Path) -> Dict[str, Dict[str, str]]:
    label_dict: Dict[str, Dict[str, str]] = {}

    with label_txt_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            name, class_id, species, breed_id = parts[:4]
            label_dict[name] = {
                "class_id": class_id,
                "species": species,   # 1: cat, 2: dog
                "breed_id": breed_id,
            }

    return label_dict


def _resolve_image_path(image_dir: Path, name: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{name}{ext}"
        if path.exists():
            return path
    return None


def _parse_single_bbox(xml_path: Path) -> Optional[List[int]]:
    """
    단일 객체만 사용한다. 객체가 1개가 아니면 None 반환.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        objects = root.findall("object")
        if len(objects) != 1:
            return None

        bndbox = objects[0].find("bndbox")
        if bndbox is None:
            return None

        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        if xmin >= xmax or ymin >= ymax:
            return None

        return [xmin, ymin, xmax, ymax]

    except Exception:
        return None


def _build_trainval_samples() -> List[dict]:
    p = get_paths()
    label_dict = _read_label_dict(p.trainval_label_path)

    samples: List[dict] = []
    for name, info in label_dict.items():
        img_path = _resolve_image_path(p.image_dir, name)
        xml_path = p.bbox_dir / f"{name}.xml"

        if img_path is None or not xml_path.exists():
            continue

        bbox = _parse_single_bbox(xml_path)
        if bbox is None:
            continue

        try:
            label = int(info["species"])
        except ValueError:
            continue

        # background=0, cat/dog=1/2 가정
        if label not in (1, 2):
            continue

        samples.append(
            {
                "name": name,
                "image_path": str(img_path),
                "bbox": bbox,
                "label": label,
            }
        )

    samples.sort(key=lambda x: x["name"])
    return samples


def _build_test_samples() -> List[dict]:
    """
    Oxford Pet의 test split은 xml bbox가 없는 샘플이 많으므로
    test 로더는 라벨 기반 분류 평가용 샘플을 별도로 구성한다.
    """
    p = get_paths()
    label_dict = _read_label_dict(p.test_label_path)

    samples: List[dict] = []
    for name, info in label_dict.items():
        img_path = _resolve_image_path(p.image_dir, name)
        if img_path is None:
            continue

        try:
            label = int(info["species"])
        except ValueError:
            continue

        if label not in (1, 2):
            continue

        samples.append(
            {
                "name": name,
                "image_path": str(img_path),
                "label": label,
            }
        )

    samples.sort(key=lambda x: x["name"])
    return samples


class CatDogDetectionDataset(Dataset):
    def __init__(self, samples: List[dict], transform=None):
        self.samples = samples
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> List[int]:
        return [s["label"] for s in self.samples]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)

        boxes = torch.tensor([sample["bbox"]], dtype=torch.float32)   # [1,4]
        labels = torch.tensor([sample["label"]], dtype=torch.int64)   # [1]

        # 기존 코드 호환(bbox/label) + torchvision detection 표준(boxes/labels)
        target = {
            "bbox": boxes,
            "label": labels,
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image, target


def detection_collate_fn(batch):
    return tuple(zip(*batch))


class CatDogTestDataset(Dataset):
    def __init__(self, samples: List[dict], transform=None):
        self.samples = samples
        self.transform = transform or v2.Compose([
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def labels(self) -> List[int]:
        return [s["label"] for s in self.samples]

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(sample["label"], dtype=torch.int64)
        return image, label, sample["name"]


def test_collate_fn(batch):
    return tuple(zip(*batch))


def get_train_val_loaders(
    batch_size: int = 64,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    transform=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    modeling.ipynb 전용: train/val만 반환
    """
    samples = _build_trainval_samples()
    if not samples:
        raise RuntimeError("trainval 샘플이 0개입니다. 데이터 경로와 라벨 파일을 확인하세요.")

    dataset = CatDogDetectionDataset(samples, transform=transform)
    indices = list(range(len(dataset)))

    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=dataset.labels,
        )
    except ValueError:
        # 클래스 개수 부족 등 stratify 실패 fallback
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_ratio,
            random_state=seed,
            stratify=None,
        )

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
    )

    return train_loader, val_loader


def get_test_loader(
    batch_size: int = 64,
    num_workers: int = 0,
    transform=None,
) -> DataLoader:
    """
    inference.ipynb 전용: test만 반환
    """
    samples = _build_test_samples()
    if not samples:
        raise RuntimeError("test 샘플이 0개입니다. 데이터 경로와 라벨 파일을 확인하세요.")

    dataset = CatDogTestDataset(samples, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_collate_fn,
    )


def get_metadata(val_ratio: float = 0.2, seed: int = 42) -> dict:
    """
    데이터셋 요약 정보 확인용.
    """
    trainval_samples = _build_trainval_samples()
    test_samples = _build_test_samples()

    trainval_labels = [s["label"] for s in trainval_samples]
    indices = list(range(len(trainval_samples)))

    if len(indices) > 0:
        try:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=seed,
                stratify=trainval_labels,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_ratio,
                random_state=seed,
                stratify=None,
            )
    else:
        train_idx, val_idx = [], []

    def _count(samples: List[dict]) -> Dict[int, int]:
        out = {1: 0, 2: 0}
        for s in samples:
            out[s["label"]] = out.get(s["label"], 0) + 1
        return out

    train_samples = [trainval_samples[i] for i in train_idx]
    val_samples = [trainval_samples[i] for i in val_idx]

    return {
        "trainval_total": len(trainval_samples),
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "train_label_dist": _count(train_samples),
        "val_label_dist": _count(val_samples),
        "test_label_dist": _count(test_samples),
        "classes": {0: "background", 1: "cat", 2: "dog"},
    }


if __name__ == "__main__":
    p = get_paths(ensure_dirs=True)
    m = get_model_paths()

    train_loader, val_loader = get_train_val_loaders()
    test_loader = get_test_loader()
    meta = get_metadata()

    print("root_dir :", p.root_dir)
    print("model_dir:", p.model_dir)
    print("model paths:", m)
    print("train batches:", len(train_loader))
    print("val batches  :", len(val_loader))
    print("test batches :", len(test_loader))
    print("metadata:", meta)
