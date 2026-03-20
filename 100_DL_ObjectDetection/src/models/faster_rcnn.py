import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_faster_rcnn_model(num_classes: int):
    """
    Faster R-CNN ResNet50 FPN 기반의 객체 탐지 모델 생성 함수.

    num_classes: 배경 포함 클래스 개수
        예) 알약 종류 10개라면 → num_classes = 11 (배경 1 + 알약 10)
    """
    # COCO pretrained weights 사용
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"  # torchvision>=0.13 기준
    )

    # 기존 헤드의 입력 피처 수 확인
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 새로운 헤드로 교체 (우리 프로젝트 클래스 개수에 맞게)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

