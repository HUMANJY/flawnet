import joblib
import pandas as pd
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 1. 2단계에서 저장한 "AI 뇌(Brain)" 모델 불러오기
MODEL_PATH = r"C:\Users\jjuyu\OneDrive\바탕 화면\FLAWNET_Project\model_brain.pkl"

try:
    model_brain = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"오류: {MODEL_PATH} 파일을 찾을 수 없습니다.")
    print("먼저 2단계(2_model_training.py) 코드를 실행해 훈련된 모델을 만들어주세요.")
    exit()

print(f"'{MODEL_PATH}' 모델을 성공적으로 불러왔습니다.")

# 2. "AI 눈(Eyes)" 모델이 사진을 분석했다고 "가정"한, 새로운 데이터 생성
# (이 부분이 나중에 AI "눈" 모델의 Output으로 대체되어야 합니다)
# 예시 1: 아반떼 CN7, 범퍼 수리/펜더 판금 (sc-1041841.json 기반)
new_case_1 = {
    'car_model': '아반떼(20)-CN7',
    'mileage_km': 36902,
    'num_exchange': 0,
    'num_repair': 1,      # '수리' 1건
    'num_sheet_metal': 1, # '판금' 1건
    'num_painting': 1,    # '도장' 1건
    'num_detach': 5
}

# 예시 2: 셀토스, 프런트도어 판금 (sc-1041817.json 기반)
new_case_2 = {
    'car_model': '셀토스(19)',
    'mileage_km': 13211,
    'num_exchange': 0,
    'num_repair': 0,
    'num_sheet_metal': 2, # '판금' 2건
    'num_painting': 2,    # '도장' 2건
    'num_detach': 4
}

# 3. AI가 인식할 수 있는 DataFrame 형태로 변환
new_data = pd.DataFrame([new_case_1, new_case_2])

# 4. "AI 뇌(Brain)" 모델로 최종 견적 "예측"
predicted_costs = model_brain.predict(new_data)

# 5. 최종 결과 출력
print("\n--- FLAWNET AI 최종 견적 예측 결과 ---")

print(f"\n[Case 1: {new_case_1['car_model']} 판금/수리]")
print(f"-> AI 예측 총 수리비: 약 {int(predicted_costs[0]):,} 원")
print(f"  (참고: 원본 견적서 청구액: 515,306 원)")

print(f"\n[Case 2: {new_case_2['car_model']} 도어 2곳 판금]")
print(f"-> AI 예측 총 수리비: 약 {int(predicted_costs[1]):,} 원")
print(f"  (참고: 원본 견적서 청구액: 585,794 원)")