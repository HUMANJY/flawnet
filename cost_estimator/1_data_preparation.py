import json
import glob
import os
import pandas as pd
import re
from tqdm import tqdm

# 1. 데이터 경로 설정 (사용자님 경로로 수정 완료)
# (Raw string r"..."를 사용하여 띄어쓰기 오류 방지)
JSON_FOLDER_PATH = r"C:\Users\jjuyu\OneDrive\바탕 화면\FLAWNET_Project\data\TS_99. 붙임_견적서"
OUTPUT_CSV_PATH = r"C:\Users\jjuyu\OneDrive\바탕 화면\FLAWNET_Project\data\brain_training_data.csv"

# 2. 샘플링 설정 (테스트 시 5000, 실제 학습 시 -1)
# 12만개 전체를 처리하려면 1~2시간 이상 걸릴 수 있습니다.
# 먼저 5000개로 테스트 후, 잘 되면 -1로 바꾸세요.
SAMPLE_SIZE = -1 
# SAMPLE_SIZE = -1 # (전체 데이터로 학습 시)

print(f"'{JSON_FOLDER_PATH}' 경로에서 JSON 파일 검색을 시작합니다...")
json_files = glob.glob(os.path.join(JSON_FOLDER_PATH, '**', '*.json'), recursive=True)

if not json_files:
    print(f"경고: '{JSON_FOLDER_PATH}'에서 JSON 파일을 찾을 수 없습니다.")
    print("경로를 다시 확인해주세요.")
else:
    print(f"총 {len(json_files)}개의 견적서 JSON 파일을 찾았습니다.")
    
    # 3. 샘플링 적용
    if SAMPLE_SIZE > 0:
        print(f"이 중 {min(len(json_files), SAMPLE_SIZE)}개를 샘플링하여 처리합니다.")
        sample_files = json_files[:min(len(json_files), SAMPLE_SIZE)]
    else:
        print(f"전체 데이터 {len(json_files)}개를 처리합니다...")
        sample_files = json_files

    # 4. JSON 파싱 -> AI 학습용 데이터로 변환
    invoice_records = []
    num_extractor = re.compile(r'\d+')

    for file_path in tqdm(sample_files, desc="JSON 파일 처리 중"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            total_cost_str = data['수리비 정산정보']['합계'].get('청구액', '0')
            total_cost = int("".join(num_extractor.findall(total_cost_str)))
            
            if total_cost == 0: continue

            car_model = data['차량정보'].get('차량명칭', 'Unknown')
            mileage_str = data['차량정보'].get('주행거리', '0Km')
            mileage = int("".join(num_extractor.findall(mileage_str)))

            operations = [item.get('작업', '').strip() for item in data['수리내역']]
            num_exchange = operations.count('교환')
            num_repair = operations.count('수리')
            num_sheet_metal = operations.count('판금')
            num_painting = operations.count('도장')
            num_detach = operations.count('탈착')

            record = {
                'car_model': car_model,
                'mileage_km': mileage,
                'num_exchange': num_exchange,
                'num_repair': num_repair,
                'num_sheet_metal': num_sheet_metal,
                'num_painting': num_painting,
                'num_detach': num_detach,
                'Y_TOTAL_COST_WON': total_cost 
            }
            invoice_records.append(record)
        
        except Exception as e:
            pass 

    # 5. 최종 결과를 CSV 파일로 저장
    if invoice_records:
        df = pd.DataFrame(invoice_records)
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        
        print(f"\n--- AI '뇌(Brain)' 모델 학습용 교과서(CSV) 생성 완료! ---")
        print(f"총 {len(df)}개의 유효한 견적 데이터를 '{OUTPUT_CSV_PATH}' 파일로 저장했습니다.")
    else:
        print("처리할 수 있는 유효한 견적 데이터가 없습니다.")