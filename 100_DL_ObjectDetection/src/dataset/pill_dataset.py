import os
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


class PillDataset(Dataset):
    """
    경구약제 객체 탐지용 PyTorch Dataset 템플릿.

    [가정]
    - 주어진 라벨 CSV (또는 DataFrame)는 대략 아래와 같은 컬럼을 가진다고 가정:
        image_id, xmin, ymin, xmax, ymax, label
    - 같은 image_id가 여러 행에 등장할 수 있음 (한 이미지에 여러 알약).

    실제 Competition 라벨 형식에 맞게
    __init__ / __getitem__ 부분을 수정하면 됨.
    """

    def __init__(
        self,
        img_dir: str,
        annotations,
        transforms=None,
    ) -> None:
        """
        img_dir: 이미지가 저장된 디렉토리 경로
        annotations: Pandas DataFrame 또는 그와 유사한 객체
                     (image_id, xmin, ymin, xmax, ymax, label 포함)
        transforms: 이미지/타깃에 적용할 transforms (Albumentations or torchvision)
        """
        self.img_dir = img_dir
        self.annotations = annotations
        self.transforms = transforms

        # image_id 리스트 (중복 제거)
        self.image_ids = sorted(self.annotations["image_id"].unique())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        image_id = self.image_ids[idx]

        # 해당 이미지에 대한 라벨만 필터링
        records = self.annotations[self.annotations["image_id"] == image_id]

        # 이미지 로드 (PIL → RGB)
        img_path = os.path.join(self.img_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        # 박스, 라벨 텐서화
        boxes: List[List[float]] = []
        labels: List[int] = []

        for _, row in records.iterrows():
            boxes.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            labels.append(int(row["label"]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # area, iscrowd 등 (COCO 스타일)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target: Dict[str, Any] = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
        }

        # transforms가 Albumentations 등이라면
        if self.transforms is not None:
            # Albumentations 예시:
            # transformed = self.transforms(image=np.array(image), bboxes=boxes, labels=labels)
            # image = transformed["image"]
            # boxes  = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            # target["boxes"] = boxes
            pass  # TODO: 실제 transforms 로직으로 교체

        return image, target
