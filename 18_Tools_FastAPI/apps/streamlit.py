import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"


def fetch_movies() -> list[dict]:
    resp = requests.get(f"{BACKEND_URL}/movies")
    resp.raise_for_status()
    return resp.json()


st.title("영화 정보 서비스")

# ── 영화 등록 ────────────────────────────────────────────
st.header("영화 등록")
with st.form("add_movie"):
    title = st.text_input("제목")
    release_date = st.date_input("개봉일")
    director = st.text_input("감독")
    genre = st.text_input("장르")
    poster_url = st.text_input("포스터 URL")
    submitted = st.form_submit_button("등록")

if submitted:
    if not all([title, director, genre, poster_url]):
        st.error("모든 항목을 입력해주세요.")
    else:
        resp = requests.post(f"{BACKEND_URL}/movies", json={
            "title": title,
            "release_date": str(release_date),
            "director": director,
            "genre": genre,
            "poster_url": poster_url,
        })
        if resp.status_code == 201:
            st.success(f"'{title}' 등록 완료!")
        else:
            st.error(f"등록 실패: {resp.text}")

# ── 영화 목록 ────────────────────────────────────────────
st.header("영화 목록")
try:
    movies = fetch_movies()
except Exception as e:
    st.error(f"백엔드 연결 실패: {e}")
    movies = []

if not movies:
    st.info("등록된 영화가 없습니다.")
else:
    cols = st.columns(3)
    for i, movie in enumerate(movies):
        with cols[i % 3]:
            if movie["poster_url"]:
                st.image(movie["poster_url"], use_container_width=True)
            st.subheader(movie["title"])
            st.caption(f"{movie['release_date']} | {movie['genre']} | {movie['director']}")
            if st.button("삭제", key=f"del_{movie['id']}"):
                resp = requests.delete(f"{BACKEND_URL}/movies/{movie['id']}")
                if resp.status_code == 204:
                    st.success("삭제 완료!")
                    st.rerun()
                else:
                    st.error(f"삭제 실패: {resp.text}")
