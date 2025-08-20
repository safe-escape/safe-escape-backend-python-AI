# train.py
import os, json, joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTENC

RANDOM_STATE = 42
TEST_SIZE = 0.2

# DATA_DENSITY = os.getenv("DENSITY_CSV", "density.csv")
# DATA_CONGESTION = os.getenv("CONGESTION_CSV", "congestion.csv")
DATA_DENSITY = "./density.csv"
DATA_CONGESTION = "./congestion.csv"
MODEL_DIR = "./model"
MODEL_PATH = os.path.join(MODEL_DIR, "lgbm.pkl")
META_PATH  = os.path.join(MODEL_DIR, "meta.json")

# LightGBM 하이퍼파라미터
LGBM_KW = dict(
    objective="multiclass",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=RANDOM_STATE
)
# 범주형 컬럼 category dtype으로 변환
def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["weekday", "holiday", "location"]:
        df[c] = df[c].astype("category")
    return df

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_raw = pd.read_csv(DATA_DENSITY)                  # timestamp, weekday, holiday, location
    y_raw = pd.read_csv(DATA_CONGESTION)["congestion_level"]    #레이블

    df = X_raw.copy()
    df["congestion_level"] = y_raw
    df = df.dropna().copy()
    for c in ["timestamp", "weekday", "holiday", "location", "congestion_level"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")   #숫자형으로 변환
    df = df.dropna().copy()
    df["congestion_level"] = df["congestion_level"].astype(int) 

    X = df[["timestamp", "weekday", "holiday", "location"]].copy()
    y = df["congestion_level"].copy()

    #전체 location, label목록
    locations = sorted(df["location"].astype(int).unique().tolist())
    target_labels = sorted(y.unique().tolist())

    # train, valid 데이터 분할(label 0~3의 분포가 극단적으로 불균형적이므로 비율 맞춰서 분할)
    stratify_ok = y.nunique() > 1 and y.value_counts().min() >= 2
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=(y if stratify_ok else None)
    )

    # SMOTENC 샘플링 기법 활용(레이블 클래스 불균형 완화)
    cat_idx = [1, 2, 3]
    smote = SMOTENC(categorical_features=cat_idx, sampling_strategy="auto",
                    random_state=RANDOM_STATE, n_jobs=-1)

    X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr) #훈련셋에만 오버샘플링 수행

    # LGBM 범주형 처리 위해 Category로 캐스팅
    X_tr_res = to_categorical(pd.DataFrame(X_tr_res, columns=X.columns))
    X_te_cat  = to_categorical(X_te)

    # LGBM 학습
    clf = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=y.nunique(),
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=1,         
    num_threads=1
    )
    
    clf.fit(
        X_tr_res, y_tr_res,
        eval_set=[(X_te_cat, y_te)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    # 평가
    y_pred = clf.predict(X_te_cat)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    print(f"[Eval] Acc={acc:.4f} | MacroF1={f1m:.4f}")
    print(classification_report(y_te, y_pred, digits=4))
    model_to_save = clf

    # 저장
    joblib.dump(model_to_save, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "locations": locations,
                "target_labels": target_labels,
                "feature_order": ["timestamp", "weekday", "holiday", "location"]
            },
            f,
            ensure_ascii=False,
            indent=2
        )
    print(f"[Saved] model -> {MODEL_PATH}")
    print(f"[Saved] meta  -> {META_PATH}")

if __name__ == "__main__":
    main()
