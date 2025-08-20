import requests
import pandas as pd
import os
import urllib.parse
from datetime import datetime
import xml.etree.ElementTree as ET

# 혼잡도 관련 매핑
hotspot_name = {
    'DDP(동대문디자인플라자)': 0, 'DMC(디지털미디어시티)': 1,'가락시장': 2, '가로수길': 3,
    '가산디지털단지역': 4, '강남 MICE 관광특구': 5, '강남역': 6, '강서한강공원': 7,
    '건대입구역': 8, '경복궁': 9, '고덕역': 10, '고속터미널역': 11, 
    '고척돔': 12, '광나루한강공원': 13, '광장(전통)시장': 14, '광화문·덕수궁': 15,
    '광화문광장': 16, '교대역': 17, '구로디지털단지역': 18, '구로역': 19,
    '국립중앙박물관·용산가족공원': 20, '군자역': 21, '김포공항': 22, '난지한강공원': 23,
    '남대문시장': 24, '남산공원': 25, '노들섬': 26, '노량진': 27,
    '대림역': 28, '덕수궁길·정동길': 29, '동대문 관광특구': 30, '동대문역': 31,
    '뚝섬역': 32, '뚝섬한강공원': 33, '망원한강공원': 34, '명동 관광특구': 35,
    '미아사거리역': 36, '반포한강공원': 37, '발산역': 38, '보라매공원': 39,
    '보신각': 40, '북서울꿈의숲': 41, '북창동 먹자골목': 42, '북촌한옥마을': 43,
    '사당역': 44, '삼각지역': 45, '서대문독립공원': 46, '서리풀공원·몽마르뜨공원': 47,
    '서울 암사동 유적': 48, '서울광장': 49, '서울대공원': 50, '서울대입구역': 51,
    '서울숲공원': 52, '서울식물원·마곡나루역': 53, '서울역': 54, '서촌': 55,
    '선릉역': 56, '성수카페거리': 57, '성신여대입구역': 58, '송리단길·호수단길': 59,
    '수유역': 60, '신논현역·논현역': 61, '신도림역': 62, '신림역': 63,
    '신정네거리역': 64, '신촌 스타광장': 65, '신촌·이대역': 66, '쌍문역': 67,
    '아차산': 68, '안양천': 69, '압구정로데오거리': 70, '양재역': 71,
    '양화한강공원': 72, '어린이대공원': 73, '여의도': 74, '여의도한강공원': 75,
    '여의서로': 76, '역삼역': 77, '연남동': 78, '연신내역': 79,
    '영등포 타임스퀘어': 80, '오목교역·목동운동장': 81, '올림픽공원': 82, '왕십리역': 83,
    '용리단길': 84, '용산역': 85, '월드컵공원': 86, '응봉산': 87,
    '이촌한강공원': 88, '이태원 관광특구': 89, '이태원 앤틱가구거리': 90, '이태원역': 91,
    '익선동': 92, '인사동': 93, '잠실 관광특구': 94, '잠실롯데타워 일대': 95,
    '잠실새내역': 96, '잠실역': 97, '잠실종합운동장': 98, '잠실한강공원': 99,
    '잠원한강공원': 100, '장지역': 101, '장한평역': 102, '종로·청계 관광특구': 103,
    '창덕궁·종묘': 104, '창동 신경제 중심지': 105, '천호역': 106, '청계산': 107,
    '청담동 명품거리': 108, '청량리 제기동 일대 전통시장': 109, '청와대': 110, '총신대입구(이수)역': 111,
    '충정로역': 112, '합정역': 113, '해방촌·경리단길': 114, '혜화역': 115,
    '홍대 관광특구': 116, '홍대입구역(2호선)': 117, '홍제폭포': 118, '회기역': 119    
}

congestion_map = {
    '여유': 0,
    '보통': 1,
    '약간 붐빔': 2,
    '붐빔': 3,
}

#공휴일 유/무 판단 api 호출
def req_pub_holiday(dt_tm):
    SERV_KEY = 'sYYjacIiF2DFsJ/354X/Vv7qGQ1sKHdnQSdrd8sT/MjWbp38pHzwL4q/LgRd/bK2wXlzmgPRQiS3uqTGdd/Bcw=='
    year = dt_tm.year
    month = dt_tm.month
    day = dt_tm.day
    url = 'http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getAnniversaryInfo'
    params = {
        'serviceKey': SERV_KEY,
        'pageNo': '1',
        'numOfRows': '10',
        'solYear': str(year),
        'solMonth': f'{month:02d}'
    }

    response = requests.get(url, params=params)
    xml_str = response.content.decode("utf-8")
    root = ET.fromstring(xml_str)

    target_date_str = f'{year}{month:02d}{day:02d}'

    for item in root.findall(".//item"):
        locdate_elem = item.find('locdate')
        if locdate_elem is not None and locdate_elem.text == target_date_str:
            return 1
    return 0

# 혼잡도 api 필요 데이터 추출 함수
def fetch_congestion_info(root):
    try:
        info = root["SeoulRtd.citydata_ppltn"][0]
        s = info["PPLTN_TIME"]
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
        weekday = dt.weekday()
        is_holiday = req_pub_holiday(dt)
        hr = dt.hour

        return {
            "timestamp": hr,
            "weekday": weekday,
            "holiday": is_holiday,
            "location": hotspot_name[info["AREA_NM"]],
            "congestion_level": congestion_map[info["AREA_CONGEST_LVL"]]
        }
    except (KeyError, IndexError) as e:
        raise Exception(f"JSON 구조 오류: {e}")

def append_to_csv(row_dict, csv_path):
    new_row = pd.DataFrame([row_dict])
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row
    df.to_csv(csv_path, index=False)
    print(f"✅ 저장 완료: {row_dict}")

def edit_csv():
    for i in hotspot_name:
        try:
            API_KEY = "6d656141446b6a3639336c704d756b"
            encoded_i = urllib.parse.quote(i)
            url = f"http://openapi.seoul.go.kr:8088/{API_KEY}/json/citydata_ppltn/1/5/{encoded_i}"
            print("요청 URL:", url)

            response = requests.get(url)
            if response.status_code != 200:
                print(f"요청 실패: {response.status_code}")
                continue

            try:
                root = response.json()
            except ValueError:
                print(f"JSON 파싱 실패: {response.text}")
                continue

            row = fetch_congestion_info(root)

            den_row = {
                "timestamp": row["timestamp"],
                "weekday": row["weekday"],
                "holiday": row["holiday"],
                "location": row["location"]
            }
            cong_row = {
                "congestion_level": row["congestion_level"]
            }

            append_to_csv(den_row, "/home/ec2-user/density.csv")
            append_to_csv(cong_row, "/home/ec2-user/congestion.csv")

        except Exception as e:
            print(f"{i} 처리 중 오류 발생: {e}")

if __name__ == '__main__':
    edit_csv()
