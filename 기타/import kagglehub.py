import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8')

liked = pd.read_csv("high_popularity_spotify_data.csv")
hated = pd.read_csv("low_popularity_spotify_data.csv")

# 쓸 칼럼: track_popularity, track_album_release_date, duration_ms
pop = "track_popularity"
year = "track_album_release_date"
len = "duration_ms"

# 전처리
liked[len] = (((liked[len] / 1000) // 60 ) * 60).astype(int)
liked[year] = liked[year].str[:4].astype(int)
liked[pop] = (liked[pop] // 5) * 5

sns.histplot(data = liked, x=year, hue=len, kde=False, stat="percent", 
             common_norm=False, multiple="fill", alpha=0.9)

plt.show()

# 길이에 따른 인기도 분석: liked와 hated
