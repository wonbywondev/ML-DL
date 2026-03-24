"""
TMDB API에서 한국어 기준 인기 영화 30개를 가져와 data/movies.json으로 저장합니다.
"""

import json
import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
TMDB_TOKEN = os.getenv("TMDB_TOKEN")

POSTER_BASE = "https://image.tmdb.org/t/p/w500"
TARGET = 30

GENRE_MAP = {
    28: "액션", 12: "어드벤처", 16: "애니메이션", 35: "코미디",
    80: "범죄", 99: "다큐멘터리", 18: "드라마", 10751: "가족",
    14: "판타지", 36: "역사", 27: "공포", 10402: "음악",
    9648: "미스터리", 10749: "로맨스", 878: "SF", 10770: "TV 영화",
    53: "스릴러", 10752: "전쟁", 37: "서부",
}


def fetch_popular(page: int) -> list[dict]:
    headers = {"Authorization": f"Bearer {TMDB_TOKEN}"}
    r = requests.get(
        "https://api.themoviedb.org/3/movie/popular",
        headers=headers,
        params={"language": "ko-KR", "page": page},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()["results"]


def to_movie(raw: dict) -> dict | None:
    if not raw.get("poster_path") or not raw.get("release_date"):
        return None
    genres = [GENRE_MAP[g] for g in raw.get("genre_ids", []) if g in GENRE_MAP]
    return {
        "title": raw["title"],
        "release_date": raw["release_date"],
        "director": "",
        "genre": genres[0] if genres else "기타",
        "poster_url": POSTER_BASE + raw["poster_path"],
        "_tmdb_id": raw["id"],
    }


def fetch_director(tmdb_id: int) -> str:
    headers = {"Authorization": f"Bearer {TMDB_TOKEN}"}
    r = requests.get(
        f"https://api.themoviedb.org/3/movie/{tmdb_id}/credits",
        headers=headers,
        params={"language": "ko-KR"},
        timeout=10,
    )
    r.raise_for_status()
    crew = r.json().get("crew", [])
    directors = [p["name"] for p in crew if p["job"] == "Director"]
    return directors[0] if directors else "미상"


def crawl(target: int = TARGET) -> list[dict]:
    movies, page = [], 1
    while len(movies) < target:
        for raw in fetch_popular(page):
            m = to_movie(raw)
            if m is None:
                continue
            m["director"] = fetch_director(m.pop("_tmdb_id"))
            movies.append(m)
            print(f"  [{len(movies):02d}] {m['title']} / {m['director']}")
            if len(movies) >= target:
                break
        page += 1
    return movies[:target]


if __name__ == "__main__":
    out_path = Path(__file__).parent.parent / "data" / "movies.json"
    out_path.parent.mkdir(exist_ok=True)

    print(f"TMDB에서 영화 {TARGET}개 수집 중...")
    movies = crawl(TARGET)
    out_path.write_text(json.dumps(movies, ensure_ascii=False, indent=2))
    print(f"\n완료: {out_path} ({len(movies)}개)")
