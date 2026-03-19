import pytest
from fastapi.testclient import TestClient
import apps.backend as backend
from apps.backend import app


@pytest.fixture(autouse=True)
def reset_db():
    """각 테스트 전 DB 초기화"""
    backend.movies.clear()
    backend.next_id = 1
    yield


client = TestClient(app)

MOVIE_PAYLOAD = {
    "title": "인터스텔라",
    "release_date": "2014-11-06",
    "director": "크리스토퍼 놀란",
    "genre": "SF",
    "poster_url": "https://example.com/interstellar.jpg",
}


# ── 영화 등록 ────────────────────────────────────────────
def test_add_movie():
    resp = client.post("/movies", json=MOVIE_PAYLOAD)
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == 1
    assert data["title"] == "인터스텔라"


def test_add_movie_auto_increment_id():
    client.post("/movies", json=MOVIE_PAYLOAD)
    resp = client.post("/movies", json={**MOVIE_PAYLOAD, "title": "기생충"})
    assert resp.json()["id"] == 2


# ── 전체 영화 조회 ─────────────────────────────────────
def test_get_movies_empty():
    resp = client.get("/movies")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_movies():
    client.post("/movies", json=MOVIE_PAYLOAD)
    client.post("/movies", json={**MOVIE_PAYLOAD, "title": "기생충"})
    resp = client.get("/movies")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


# ── 특정 영화 조회 ─────────────────────────────────────
def test_get_movie():
    client.post("/movies", json=MOVIE_PAYLOAD)
    resp = client.get("/movies/1")
    assert resp.status_code == 200
    assert resp.json()["title"] == "인터스텔라"


def test_get_movie_not_found():
    resp = client.get("/movies/999")
    assert resp.status_code == 404


# ── 영화 삭제 ────────────────────────────────────────────
def test_delete_movie():
    client.post("/movies", json=MOVIE_PAYLOAD)
    resp = client.delete("/movies/1")
    assert resp.status_code == 204
    assert client.get("/movies/1").status_code == 404


def test_delete_movie_not_found():
    resp = client.delete("/movies/999")
    assert resp.status_code == 404
