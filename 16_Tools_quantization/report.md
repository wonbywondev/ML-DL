# 미션 16 요약 보고서
## 모델 변환 및 추론 비교 (SSD300 VGG16 · Oxford-IIIT Pet)

---

## 1. 모델 개요

| 항목 | 내용 |
|------|------|
| 모델 | SSD300 VGG16 (`torchvision.models.detection.ssd300_vgg16`) |
| 데이터셋 | Oxford-IIIT Pet Dataset |
| 클래스 | background(0), cat(1), dog(2) — 총 3개 |
| Backbone | VGG16 (pretrained on ImageNet) |
| 학습 방식 | Backbone만 사전학습, SSD Head는 랜덤 초기화 후 fine-tuning(1 epoch) |
| 옵티마이저 | SGD (lr=1e-4, momentum=0.9, weight_decay=5e-4) |
| 스케줄러 | StepLR (step_size=2, gamma=0.1) |

### 데이터셋 샘플 이미지

| Cat (Abyssinian) | Dog (American Bulldog) |
|:---:|:---:|
| ![cat sample](assets/sample_cat.jpg) | ![dog sample](assets/sample_dog.jpg) |

---

## 2. 추출된 모델 파일 및 용량

| 파일명 | 형식 | 용량 |
|--------|------|------|
| `mission_16_ssd300_vgg16.pth` | PyTorch FP32 | 91 MB |
| `mission_16_ssd300_vgg16_quantized.pth` | PyTorch Quantized (INT8) | 91 MB |
| `mission_16_ssd300_vgg16.onnx` | ONNX | 91 MB |

> **참고:** Dynamic Quantization은 `torch.nn.Linear` 레이어에만 INT8 적용(`torch.ao.quantization.quantize_dynamic`). SSD300의 핵심 연산이 Conv2d 위주이므로 파일 크기 감소 효과가 미미함.

---

## 3. 추론 성능 비교 결과

평가 데이터셋: Oxford-IIIT Pet **test split** (3,669장)
평가 지표: Top-1 Accuracy (가장 높은 confidence score의 predicted label 기준)

| 모델 | Accuracy | 총 소요 시간 | 이미지당 추론 시간 |
|------|----------|-------------|-------------------|
| PyTorch FP32 | 52.88% | 1,279.99 초 | 348.87 ms |
| PyTorch Quantized (INT8) | 52.88% | 1,292.04 초 | 352.15 ms |
| ONNX (CPUExecutionProvider) | 52.88% | 1,102.50 초 | 300.49 ms |

**분석:**
- 세 모델 모두 동일한 Accuracy(52.88%)를 보임 — 양자화로 인한 정확도 손실 없음
- ONNX가 FP32 대비 **약 14% 빠른 추론 속도** (348.87 ms → 300.49 ms)
- Quantized PyTorch는 FP32보다 오히려 소폭 느림 — Linear 레이어 비중이 작아 Dynamic Quantization 오버헤드가 이득보다 큰 것으로 추정

---

## 4. 코드 구성 — `preparation.py` 별도 모듈화

### 배경: 학습 데이터와 테스트 데이터의 구조 차이

Oxford-IIIT Pet 데이터셋은 공식 annotation이 split별로 다르게 제공된다.

```
annotations/
├── trainval.txt   ← 이미지명, 클래스, species, breed_id
├── test.txt       ← 이미지명, 클래스, species, breed_id
└── xmls/          ← bbox XML (trainval용 3,686개만 존재)
```

선행 미션에서 이미 split별 현황을 집합 연산으로 확인한 결과:

```
[train 데이터 오류 확인]
· 이미지 O | label O | 좌표 O: 3,671개   ← 정상 학습 가능
· 이미지 X | label O | 좌표 X:     9개   ← bbox·이미지 누락, 제외

[test 데이터 오류 확인]
· 이미지 O | label O | 좌표 O:     0개   ← bbox 보유 샘플 없음
· 이미지 X | label O | 좌표 X: 3,669개   ← label만 존재
```

**test split 전체에 bbox XML이 존재하지 않는다.** 따라서 `inference.ipynb`에서 학습용 Dataset(`CatDogDetectionDataset`)을 그대로 쓰면 XML 파싱 단계에서 전체 샘플이 누락되어 테스트가 불가능하다.

### 결과: Dataset 클래스 분리 불가피

| 구분 | `CatDogDetectionDataset` | `CatDogTestDataset` |
|------|--------------------------|---------------------|
| 사용 split | `trainval.txt` | `test.txt` |
| 필요 정보 | 이미지 + bbox XML + 라벨 | 이미지 + 라벨 |
| 출력 형태 | `(image, target_dict)` | `(image, label, name)` |
| 사용 노트북 | `modeling.ipynb` | `inference.ipynb` |

두 Dataset이 공유하는 경로 계산·라벨 파싱·이미지 탐색 로직을 `preparation.py`로 추출해 양쪽 노트북이 `from preparation import ...`으로 공유하도록 구성했다.

```
notebooks/
├── preparation.py        ← 공통 모듈 (경로, Dataset, DataLoader)
├── modeling.ipynb        ← get_train_val_loaders() 사용
└── inference.ipynb       ← get_test_loader() 사용
```

---

## 5. 디버깅 및 오류 사항 정리

### modeling.ipynb

#### 5-1. DeprecationWarning: `torch.ao.quantization` deprecated

```
DeprecationWarning: torch.ao.quantization is deprecated and will be removed in 2.10.
For migrations: use torchao eager mode quantize_ API instead.
```

**원인:** PyTorch 2.x 이상에서 `torch.ao.quantization`의 eager mode API가 deprecated됨.

**해결:** 현재 미션 범위에서는 경고만 발생하며 동작에는 문제 없어 그대로 진행. 향후에는 `torchao` 패키지의 `quantize_()` API로 마이그레이션 필요.

---

#### 5-2. `torch.onnx.export` Legacy TorchScript 경고

```
DeprecationWarning: You are using the legacy TorchScript-based ONNX export.
Starting in PyTorch 2.9, the new torch.export-based ONNX exporter has become the default.
```

**원인:** `dynamo=False` 옵션으로 구 방식(TorchScript tracing) 사용.

**해결:** SSD300 모델이 동적 제어 흐름을 가지고 있어 `dynamo=True` 전환 시 추가 대응이 필요. 현재는 `opset_version=17`로 안정적으로 내보내기 완료.

---

#### 5-3. `torch.tensor(sourceTensor)` UserWarning (torchvision 내부)

```
UserWarning: To copy construct from a tensor, it is recommended to use
sourceTensor.detach().clone() rather than torch.tensor(sourceTensor).
```

**발생 위치:** `torchvision/ops/boxes.py`, `torchvision/models/detection/transform.py`

**원인:** torchvision 내부 코드에서 발생. 사용자 코드가 아니므로 직접 수정 불가.

**해결:** 동작 및 결과에는 영향 없음. 무시.

---

### inference.ipynb

#### 5-4. ONNX 출력 순서 불일치

**상황:** `output_names=["boxes", "labels", "scores"]` 순으로 내보냈으나, 실제 `session.run()` 반환값의 dtype을 확인하니 두 번째·세 번째 출력의 타입이 예상과 반대였다.

```python
# onnx outputs 확인 결과
[('boxes',  [..., 4], 'tensor(float)'),
 ('labels', ['dim_0'], 'tensor(float)'),   # ← 실제로는 scores (float)
 ('scores', ['dim_0'], 'tensor(int64)')]   # ← 실제로는 labels (int64)
```

**원인:** `torch.onnx.export`의 TorchScript tracing 과정에서 내부 출력 순서가 재배열됨.

**해결:** `session.run()` 결과를 dtype으로 순서를 역추적해 올바르게 언패킹.

```python
boxes, scores_f, labels_i = session.run(None, {input_name: x})
```

---

## 6. 결론

- Dynamic Quantization은 Conv2d 중심의 detection 모델에서 파일 크기 및 속도 이득이 제한적
- ONNX 형태가 CPU 환경에서 가장 빠른 추론 달성 (이미지당 300.49 ms, FP32 대비 14% 향상)
- 세 형태 모두 동일한 정확도(52.88%) 유지 — 변환 과정의 수치적 무손실 확인