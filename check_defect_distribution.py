import os
import json
# train.py로부터 기준이 되는 정보(파일 리스트, damage_map)를 가져옵니다.
from train import defect_files, damage_map

JSON_DIR = './data/jsons'

# 1. train.py의 damage_map을 기준으로 통계판(counts)을 만듭니다.
counts = {name: 0 for name in damage_map.keys()}
unknown_count = 0
unknown_labels = set()

# 2. 결함 파일들을 순회하며 JSON을 읽고 종류별로 카운트합니다.
for img_name in defect_files:
    json_name = img_name.replace('.jpg', '.json')
    json_path = os.path.join(JSON_DIR, json_name)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # JSON 파일에서 실제 결함 이름을 읽어옵니다. (예: 'Scratched')
        damage_str = data['annotations'][0].get('damage', 'Unknown')
        
        # 3. 데이터의 'Scratched'를 코드 기준인 'Scratch'로 변환합니다.
        if damage_str == 'Scratched':
            damage_str = 'Scratch'

        # 4. 변환된 이름이 기준 통계판(counts)에 있는지 확인합니다.
        if damage_str in counts:
            counts[damage_str] += 1
        else:
            unknown_count += 1
            unknown_labels.add(damage_str)

# --- 5. 최종 결과 출력 ---
print("--- 📊 결함 종류별 데이터 분포 ---")
for name, count in counts.items():
    print(f"- {name}: {count}개")

# 그 외 정의되지 않은 라벨이 있다면 경고 메시지를 보여줍니다.
if unknown_count > 0:
    print("\n[주의!] train.py의 damage_map에 정의되지 않은 라벨을 발견했습니다.")
    print(f"발견된 라벨: {list(unknown_labels)} ({unknown_count}개)")