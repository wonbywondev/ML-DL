# 기본 세팅
import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(DATA_DIR, exist_ok=True)

train_csv_path = os.path.join(RAW_DIR, "mission15_train.csv")
test_csv_path = os.path.join(RAW_DIR, "mission15_test.csv")

train_df = pd.DataFrame(pd.read_csv(train_csv_path))
test_df = pd.DataFrame(pd.read_csv(test_csv_path))


# 전처리
import copy

    # 중복치 제거
train_df = train_df.drop_duplicates()

    # 쓸모 없는 float -> int로 변환
train_df["Performance Index"] = train_df["Performance Index"].astype(int)

    # str(y/n) -> int로 변환
def yn_to_int(df: pd.DataFrame):    
    df["Extracurricular Activities"] = (
        df["Extracurricular Activities"].replace({"No": "0", "Yes": "1"})
    ).astype(int)

    return df

train_df = yn_to_int(train_df)
test_df = yn_to_int(test_df)

TRAIN_DF = copy.deepcopy(train_df)
TEST_DF = copy.deepcopy(test_df)


# 학습 데이터 구성
from sklearn.model_selection import train_test_split

X_whole = TRAIN_DF.drop("Performance Index", axis=1)
y_whole = TRAIN_DF["Performance Index"]

X_train, X_val, y_train, y_val = train_test_split(X_whole, y_whole, test_size=0.2, random_state=42)

X_test = TEST_DF.copy()


# 모델링 및 저장
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error as rmse

    # 모델링
model = LinearRegression()

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

val_loss = rmse(y_val, y_val_pred)

print(f"· RMSE: {val_loss:.4f}")

    # 저장
pkl_path = os.path.join(DATA_DIR, "model", "model.pkl")
os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

with open(pkl_path, "wb") as f:
    pickle.dump(model, f)

print("· 모델 생성 완료")