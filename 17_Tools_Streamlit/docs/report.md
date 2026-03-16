  ## 모델 선택
  
  핵심 차이점

  1. MNIST / MNIST-7 / MNIST-8 / MNIST-12 — 같은 아키텍처(CNN + MaxPooling)를
  ONNX 표준이 업데이트될 때마다 새 opset으로 재export한 버전들. 성능 차이 없음.
  2. MNIST-12-int8 — MNIST-12를 INT8 양자화(quantization) 한 버전. Intel Neural
  Compressor + ONNX Runtime으로 압축해 크기가 절반(~11KB) 으로 줄어들지만
  정확도는 동일하게 유지됨.

  어떤 걸 써야 하나?

  - 일반 추론: MNIST-12 — 가장 최신 opset, 표준 float32
  - 경량화/엣지 배포: MNIST-12-int8 — 크기 절반, 속도 빠름
  - 레거시 호환성: MNIST-7 또는 MNIST-8 — 구버전 ONNX Runtime 환경


--> mnist-12-int8 선택.