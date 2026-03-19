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

> 작업 예정
