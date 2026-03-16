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

---

## 디버깅 기록

### 1. `OSError: cannot write mode F as JPEG`

**원인**
`Image.fromarray(alpha)` 에 float32 배열을 그대로 넘겨 PIL 이미지 mode가 `F`(32-bit float)로 생성됨. Streamlit이 내부적으로 JPEG으로 변환할 때 `F` 모드를 지원하지 않아 오류 발생.

**해결**
`alpha.astype(np.uint8)` 로 변환 후 `Image.fromarray()` 에 전달.

---

### 2. 숫자를 그려도 전처리 이미지·추론 결과가 갱신되지 않음

**원인**
캔버스 배경을 `#000000`(검정)으로 설정했기 때문에 초기 상태에서도 모든 픽셀의 알파값이 255. 따라서 `img_data[:, :, 3].max() > 0` 조건이 항상 True가 되어 빈 캔버스에서도 추론이 실행되거나, Streamlit 리렌더링 타이밍에 따라 결과가 들쭉날쭉하게 동작.

**해결**
- 캔버스 배경을 `#FFFFFF`(흰색), 획 색상을 `#000000`(검정)으로 변경
- "아무것도 그리지 않았는지" 판별 기준을 알파 채널 대신 `canvas_result.json_data["objects"]` 리스트 길이로 교체
- 이미지 전처리 시 흰 배경·검정 획 → MNIST 형식(검정 배경·흰 획)으로 픽셀값 반전: `255 - R채널`

---

### 3. 추론 결과가 거의 균등 분포 (모델이 숫자를 인식 못 함)

**원인**
`mnist-12-int8` 모델은 INT8 양자화 모델로, 입력으로 **0~255 범위의 float32** 값을 그대로 기대함. 그런데 코드에서 `img_array /= 255.0` 으로 0~1 범위로 정규화하여 전달했기 때문에 모든 logit이 0으로 수렴.

**검증**
```python
# 0~1 입력 → logit 전부 0
sess.run(None, {'Input3': (arr / 255.0).reshape(1,1,28,28)})
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# 0~255 입력 → 정상 logit
sess.run(None, {'Input3': arr.reshape(1,1,28,28)})
# [-1018. 1018. 1917. 59. ...]
```

**해결**
정규화 코드 제거. `img_array`를 float32로 변환만 하고 0~255 그대로 모델에 전달.

---

### 4. 막대 그래프 확률이 100%/0%로 극단적

**원인**
`mnist-12-int8` INT8 양자화 모델의 logit 스케일이 비정상적으로 큼 (spread 최대 ~14000). softmax 적용 시 항상 1위가 1.0, 나머지 0.0으로 수렴.

**시도한 해결책**
- Temperature scaling (T=2000): 확률 분포가 자연스러워지나 추론 성능 저하
- Min-max 정규화: 확률 분포가 너무 균등해짐

**최종 결론**
INT8 양자화 모델의 구조적 한계. **`mnist-12.onnx` (float32) 모델로 교체**.

---

### 5. float32 모델 교체 후에도 추론 결과 균등 (logit 전부 0)

**원인**
`@st.cache_resource`가 이전 int8 세션을 캐싱하고 있었음. 모델 경로를 `mnist-12.onnx`로 바꿔도 캐시가 살아있어 이전 int8 세션이 계속 사용됨. int8 모델은 0~1 입력에 대해 logit을 전부 0으로 반환.

**해결**
`load_model()`에 `model_path` 인자를 추가하여 캐시 키로 사용. 모델 경로가 바뀌면 자동으로 새 세션으로 재로드됨.

```python
# 변경 전: 인자 없음 → 캐시 키 고정
@st.cache_resource
def load_model() -> ort.InferenceSession: ...

# 변경 후: 경로를 캐시 키로 사용
@st.cache_resource
def load_model(model_path: str) -> ort.InferenceSession: ...

session = load_model(MODEL_PATH)
```

---

### 6. float32 모델 입력 스케일

**확인 사항**
`mnist-12.onnx` (float32) 모델은 **0~1 범위의 float32** 입력을 기대함.
- `0~255` 입력 시 logit spread ~14000 → softmax 후 1.0/0.0 극단값
- `0~1` 입력 시 logit spread ~50 → 자연스러운 확률 분포

**최종 전처리 파이프라인**
1. 캔버스 RGBA → R채널 반전 (`255 - R`) → 흰 배경·검정 획을 MNIST 형식(검정 배경·흰 글씨)으로 변환
2. 28×28 리사이즈 (PIL LANCZOS)
3. `/ 255.0` 으로 0~1 정규화
4. `(1, 1, 28, 28)` NCHW reshape → float32 → 모델 입력