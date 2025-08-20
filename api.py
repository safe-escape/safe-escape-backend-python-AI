import os, json, joblib, math
from typing import List, Optional, Dict, Literal, Tuple
from datetime import datetime
from urllib.parse import quote
import xml.etree.ElementTree as ET

from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import lightgbm as lgb
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import holidays as pyholidays

# 경로/키 설정
MODEL_PATH = "./model/lgbm.pkl"
META_PATH  = "./model/meta.json"
CSV_SHELTERS_PATH  = "./csv/shelters.csv"          # [수용시설명, 경도, 위도]
CSV_HOTSPOTS_PATH  = "./csv/population_areas.csv"  # [AREA_NM, 경도, 위도]
SEOUL_API_KEY      = "6d656141446b6a3639336c704d756b"

# 비상구 순위에 사용될 상수
HOTSPOT_RADIUS_M = 1000   # 비상구에서 1.2km 내 핫스팟의 혼잡도만 반영(도보권)
SHELTER_CAP_M    = 1000   # 비상구에서 1.2km 대피소 거리 정규화 상한(도보권)
W_CONG = 0.7              # 혼잡도 가중치(높을수록 영향 큼)
W_DIST = 0.3              # 대피소 거리 가중치

# 서울시 혼잡도 텍스트 → 등급(0~3)
CONG_MAP = {"여유": 0, "보통": 1, "약간 붐빔": 2, "붐빔": 3}

# FastAPI
app = FastAPI(
    title="Congestion + Exit Ranking API",
    description="혼잡도 예측(학습 모델)과 비상구 순위(실시간 혼잡/API, 대피소 CSV)를 제공",
    version="2.1.0",
)

# CORS 허용 미들웨어
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 아티팩트
MODEL = None    # LightGBM
META  = None    # {"locations": [...]}
SHELTERS_DF = None
HOTSPOTS_DF = None

# 공통 함수
def load_artifacts():
    """모델/메타 로드 및 CSV 로드"""
    global MODEL, META, SHELTERS_DF, HOTSPOTS_DF
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        raise RuntimeError("Model or meta file missing. Train and export artifacts first.")
    MODEL = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)
    if "locations" not in META or not isinstance(META["locations"], list):
        raise RuntimeError("META['locations'] missing or invalid.")

    if not os.path.exists(CSV_SHELTERS_PATH):
        raise RuntimeError(f"shelters csv not found: {CSV_SHELTERS_PATH}")
    if not os.path.exists(CSV_HOTSPOTS_PATH):
        raise RuntimeError(f"hotspots csv not found: {CSV_HOTSPOTS_PATH}")

    SHELTERS_DF = pd.read_csv(CSV_SHELTERS_PATH)
    HOTSPOTS_DF = pd.read_csv(CSV_HOTSPOTS_PATH)

    if "수용시설명" not in SHELTERS_DF.columns or "경도" not in SHELTERS_DF.columns or "위도" not in SHELTERS_DF.columns:
        raise RuntimeError("shelters.csv must contain ['수용시설명','경도','위도']")
    if "AREA_NM" not in HOTSPOTS_DF.columns or "경도" not in HOTSPOTS_DF.columns or "위도" not in HOTSPOTS_DF.columns:
        raise RuntimeError("population_areas.csv must contain ['AREA_NM','경도','위도']")

def compute_weekday(date_str: str) -> int:
    """월=0 ... 일=6"""
    return datetime.strptime(date_str, "%Y-%m-%d").date().weekday()

def compute_holiday_flag(date_str: str) -> int:
    """KR 법정 공휴일 여부(0/1)."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    kr_cls = getattr(pyholidays, "KR", None)
    return 1 if d in kr_cls(years=d.year) else 0

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도 좌표 간 대원거리(미터)반환 함수"""
    R = 6371000.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def fetch_congestion_levels(area_names: List[str]) -> Dict[str, Optional[int]]:
    """서울시 실시간 혼잡도 API를 핫스팟 이름별로 호출해 레벨(0~3)을 반환."""
    results: Dict[str, Optional[int]] = {name: None for name in area_names}
    if not SEOUL_API_KEY:
        return results  #실패시 None 반환
    for name in area_names:
        try:
            url = f"http://openapi.seoul.go.kr:8088/{SEOUL_API_KEY}/xml/citydata_ppltn/1/5/{quote(name)}"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            lvl_txt = root.findtext(".//AREA_CONGEST_LVL")
            if lvl_txt in CONG_MAP:
                results[name] = CONG_MAP[lvl_txt]
        except Exception:
            continue
    return results

# 라이프사이클
@app.on_event("startup")
def _startup():
    load_artifacts()

# 혼잡도 예측 df 틀 생성 함수
def build_feature_frame(date: str, hour: int, locations: List[int], holiday_override: Optional[int],) -> Tuple[pd.DataFrame, int, int]:
    if not (0 <= hour <= 23):
        raise ValueError("hour must be in 0..23.")
    weekday = compute_weekday(date)
    auto_holiday = compute_holiday_flag(date)
    holiday = holiday_override if holiday_override is not None else auto_holiday

    df = pd.DataFrame({
        "timestamp": [hour] * len(locations),
        "weekday":   [weekday] * len(locations),
        "holiday":   [holiday] * len(locations),
        "location":  locations,
    })
    for c in ["weekday", "holiday", "location"]:
        df[c] = df[c].astype("category")
    return df, weekday, holiday

# ---------- 혼잡도 예측 Pydantic 스키마 ----------
class PredictRequest(BaseModel):    #/predict 요청 바디 형식
    date: str = Field(..., example="2025-07-29", description="YYYY-MM-DD")
    hour: int = Field(..., ge=0, le=23, example=16)
    holiday_override: Optional[Literal[0, 1]] = Field(
        None, description="0=휴일 아님, 1=휴일(자동 판별 무시)"
    )
    locations: Optional[List[int]] = Field(
        None, description="비우면 META['locations'] 전체에 대해 예측"
    )
    @validator("date")
    def _check_date(cls, v):
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except Exception:
            raise ValueError("date must be YYYY-MM-DD")
        return v

class PredictionItem(BaseModel):    #/predict 예측결과
    location: int
    congestion_level: int
    proba: Optional[Dict[str, float]] = None  # 클래스별 확률(있으면)

class PredictData(BaseModel):   #/predict 응답 형식
    date: str
    hour: int
    weekday: int
    holiday: int
    n_locations: int
    predictions: List[PredictionItem]

class WrappedPredictResponse(BaseModel):    #/predict 응답 최상위(상태)
    code: Literal["OK"]
    data: PredictData

# 라우트: 혼잡도 예측
@app.post("/predict", response_model=WrappedPredictResponse)
def predict(req: PredictRequest):
    if MODEL is None or META is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    locations = req.locations if req.locations is not None else META["locations"]
    if not locations:
        raise HTTPException(status_code=400, detail="No locations provided and META['locations'] is empty")

    try:
        # 입력값 -> 특징량 DataFrame 생성 (날짜/시간/장소/휴일여부)
        feats, weekday, holiday = build_feature_frame(req.date, req.hour, locations, req.holiday_override)
        # 혼잡도 예측 모델 실행
        preds = MODEL.predict(feats)
        # proba 계산
        if hasattr(MODEL, "predict_proba"):
            P = MODEL.predict_proba(feats)  # [N, K]
            prob_list = [{str(k): float(P[i, k]) for k in range(P.shape[1])} for i in range(P.shape[0])]
        else:
            prob_list = [None] * len(preds)
    # 오류 발생 시 500 err반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    # 예측 결과 PredictionItem 리스트 변환
    items = [
        PredictionItem(location=int(locations[i]), congestion_level=int(preds[i]), proba=prob_list[i])
        for i in range(len(locations))
    ]
    # 응답 데이터 구조화
    data = PredictData(
        date=req.date,
        hour=req.hour,
        weekday=weekday,
        holiday=holiday,
        n_locations=len(locations),
        predictions=items,
    )
    return WrappedPredictResponse(code="OK", data=data)

# ---------- 비상구 순위 Pydantic 스키마 ----------
class EntranceIn(BaseModel):    #출입구 별 데이터
    id: int
    latitude: float
    longitude: float

class RankExitsRequest(BaseModel):  #/rank_exits 요청 Body 형식
    entrances: List[EntranceIn] = Field(..., description="비상구(출입구) 목록")

class RankExitsData(BaseModel): #/rank_exits 응답 형식
    ranked_entrances: List[Dict[str, float]]  # [{"id":..., "score":...}, ...]

class WrappedRankExitsResponse(BaseModel):  #/rank_exits 응답 최상위(상태)
    code: Literal["OK"]
    data: RankExitsData

# 라우트: 비상구 순위
@app.post("/rank_exits", response_model=WrappedRankExitsResponse)
def rank_exits(req: RankExitsRequest):
    if SHELTERS_DF is None or HOTSPOTS_DF is None:
        raise HTTPException(status_code=500, detail="CSV not loaded")
    # 비상구 1km 내 모든 핫스팟 집합 생성
    candidate_names: set[str] = set()
    for _, h in HOTSPOTS_DF.iterrows():
        h_lat, h_lon = float(h["위도"]), float(h["경도"])
        for ent in req.entrances:
            if haversine_m(ent.latitude, ent.longitude, h_lat, h_lon) <= HOTSPOT_RADIUS_M:
                candidate_names.add(str(h["AREA_NM"]))
                break

    #후보 핫스팟 서울시API 호출, 현재 혼잡도 획득
    cong_map = fetch_congestion_levels(sorted(candidate_names)) if candidate_names else {}

    #비상구별 점수 계산(낮을수록 좋음)
    ranked: List[Dict[str, float]] = []
    for ent in req.entrances:
        e_lat, e_lon = ent.latitude, ent.longitude
        #가장 가까운 대피소까지의 거리(정규화: 0~1, 멀수록↑)
        nearest_m = None
        for _, s in SHELTERS_DF.iterrows():
            s_lat, s_lon = float(s["위도"]), float(s["경도"])
            d_m = haversine_m(e_lat, e_lon, s_lat, s_lon)
            if (nearest_m is None) or (d_m < nearest_m):
                nearest_m = d_m
        # 대피소가 없다면 1로 처리
        dist_norm = 1.0 if nearest_m is None else min(1.0, nearest_m / float(SHELTER_CAP_M))
        #반경 내 핫스팟의 혼잡도
        worst_cong = 0
        for _, h in HOTSPOTS_DF.iterrows():
            h_nm = str(h["AREA_NM"])
            h_lat, h_lon = float(h["위도"]), float(h["경도"])
            if haversine_m(e_lat, e_lon, h_lat, h_lon) <= HOTSPOT_RADIUS_M:
                lvl = cong_map.get(h_nm, None)
                if lvl is not None:
                    worst_cong = max(worst_cong, int(lvl))
        cong_norm = worst_cong / 3.0

        # 최종 점수
        score = (dist_norm + cong_norm) / 2.0
        ranked.append({
            "id": ent.id,
            "latitude": e_lat,
            "longitude": e_lon,
            "score": round(float(score), 6)
        })
    # 오름차순 정렬
    ranked.sort(key=lambda x: x["score"])
    # 순위별로 id, 위도, 경도 반환
    output = [{"id": r["id"], "latitude": r["latitude"], "longitude": r["longitude"]} for r in ranked]
    return WrappedRankExitsResponse(
        code="OK",
        data=RankExitsData(ranked_entrances=output)
    )