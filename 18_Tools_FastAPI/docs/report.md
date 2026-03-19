# 미션 18 개발 보고서

## Step 1–3: FastAPI 백엔드 기반 구축

### 변경 사항 요약

#### 1. DB 방식 결정 — In-memory dict

| 항목 | 이전 | 이후 |
|---|---|---|
| 저장소 | `pd.read_csv()` (경로 없음, 에러) | `dict[int, dict]` in-memory |
| ID 관리 | 클라이언트가 전달 | `next_id` 카운터로 서버 자동 생성 |
| 의존성 | pandas | 없음 (표준 라이브러리만) |

in-memory dict를 선택한 이유:
- 미션 범위(학습/제출)에서 영속성 불필요
- pandas DataFrame은 row 단위 CRUD에 부적합
- SQLite보다 설정 없이 즉시 사용 가능

#### 2. Movie 모델 수정

| 필드 | 이전 | 이후 |
|---|---|---|
| `poster` | `UploadFile` (Pydantic 불가) | `poster_url: str` (URL) |
| `id` | 클라이언트가 전달 | 서버 생성, 응답 전용 |
| `rate` | `Optional[int]` (미사용) | 제거 (심화 기능으로 분리) |

`MovieCreate` (입력용) / `Movie` (응답용, id 포함) 로 분리.

#### 3. 엔드포인트 수정

| 기능 | 이전 | 이후 |
|---|---|---|
| 전체 조회 | — | `GET /movies` |
| 특정 조회 | `GET /movie/info/{movie_id}` (시그니처 오류) | `GET /movies/{movie_id}` |
| 등록 | `POST /movie/add/{movie_id}` | `POST /movies` (201) |
| 삭제 | `POST /movie/remove/{movie_id}` | `DELETE /movies/{movie_id}` (204) |

- 404 처리 추가 (존재하지 않는 id 접근 시)
- 응답 형식 `set literal` → `response_model` 기반 Pydantic 직렬화

---

## Step 4: Streamlit UI

### 구현 내용

백엔드 API를 호출하는 Streamlit 프론트엔드 작성.

#### 화면 구성

| 섹션 | 기능 |
|---|---|
| 영화 등록 | `st.form`으로 제목/개봉일/감독/장르/포스터 URL 입력 후 `POST /movies` |
| 영화 목록 | `GET /movies` 조회 후 3열 그리드로 포스터·정보 표시 |
| 삭제 버튼 | 각 카드에 삭제 버튼 → `DELETE /movies/{id}` → `st.rerun()` |

#### 설계 포인트

- 모든 데이터는 백엔드에서 관리 (Streamlit 내부 state 없음)
- `BACKEND_URL` 상수로 백엔드 주소 관리 (배포 시 환경변수로 교체 가능)
- 백엔드 연결 실패 시 에러 메시지 표시, 빈 목록으로 graceful 처리
- 포스터는 URL 기반 `st.image()` 렌더링

#### 실행 방법

```bash
# 백엔드
uvicorn apps.backend:app --reload

# 프론트엔드 (별도 터미널)
streamlit run apps/streamlit.py
```
