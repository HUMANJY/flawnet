import os
import json
from collections import defaultdict # defaultdict import 추가

# train.py에서 사용하는 4종 흠집 맵을 그대로 가져옵니다.
damage_map = {"Scratch": 0, "Crushed": 1, "Breakage": 2, "Separated": 3}

# --- ★★★ 차량 크기 분류 맵 (사용자 요청 반영) ★★★ ---
# JSON의 'supercategory_name' 값을 'small', 'mid', 'large'로 매핑합니다.
# (모두 소문자로 변환하여 비교)
size_map = {
    # small(경/소)
    "경형": "small", "소형": "small", "compact": "small", "citycar": "small",
    "mini": "small", "light": "small", "smallcar": "small", 
    "compact car": "small", "subcompact car": "small", "hatchback": "small",
    
    # mid(중형)
    "중형": "mid", "midsize": "mid", "mid": "mid", "medium": "mid",
    "intermediate": "mid", "mid-size car": "mid", "crossover": "mid",
    "sedan": "mid", "coupe": "mid",
    
    # large(대형/대형군 포함)
    "대형": "large", "large": "large", "full": "large", "fullsize": "large",
    "luxury": "large", "suv": "large", "van": "large", "wagon": "large",
    "full-size car": "large", "truck": "large", "pickup truck": "large", "bus": "large"
}
# --- ★★★★★★★★★★★★★★★★★★★★★★★★★★★ ---

def get_size_category(json_size_name):
    """JSON의 supercategory_name을 small/mid/large로 변환합니다."""
    if not json_size_name or json_size_name == "Others":
        return "알수없음"
    
    # 소문자로 변환하여 맵에서 찾음
    normalized_name = json_size_name.lower()
    
    return size_map.get(normalized_name, "알수없음")

def check_dataset_health(image_dir, json_dir, dataset_name):
    """
    지정된 이미지 및 JSON 폴더의 데이터 상태를 진단합니다.
    1. 1:1 파일 매핑 검사
    2. 4종 흠집 분포 검사
    3. 차량 크기 분포 검사
    """
    print(f"\n--- 🩺 {dataset_name} 데이터셋 진단 시작 ---")
    print(f"    Image 폴더: {image_dir}")
    print(f"    JSON 폴더: {json_dir}")
    print("-" * 40)

    try:
        image_files = set(f.replace('.jpg', '') for f in os.listdir(image_dir) if f.endswith('.jpg'))
        json_files = set(f.replace('.json', '') for f in os.listdir(json_dir) if f.endswith('.json'))
    except FileNotFoundError as e:
        print(f"[오류] 폴더를 찾을 수 없습니다: {e.filename}")
        print("경로가 './data/images/TL_image'와 같은 구조인지 확인하세요.")
        print("-" * 40)
        return

    print(f"총 이미지 파일 수 (.jpg): {len(image_files)}")
    print(f"총 JSON 파일 수 (.json): {len(json_files)}")

    # --- 1. 1:1 매핑 검사 ---
    print("\n[1:1 매핑 검사]")
    images_missing_json = list(image_files - json_files)
    jsons_missing_image = list(json_files - image_files)
    
    if not images_missing_json and not jsons_missing_image:
        print("✅ 1:1 매핑이 완벽합니다. (모든 이미지/JSON 파일이 짝을 이룸)")
    else:
        if images_missing_json:
            print(f"❌ JSON 파일이 없는 이미지: {len(images_missing_json)}개")
            if len(images_missing_json) > 5:
                print(f"  (예시: {images_missing_json[:5]} ...)")
            else:
                print(f"  (목록: {images_missing_json})")
        
        if jsons_missing_image:
            print(f"❌ 이미지 파일이 없는 JSON: {len(jsons_missing_image)}개")
            if len(jsons_missing_image) > 5:
                print(f"  (예시: {jsons_missing_image[:5]} ...)")
            else:
                print(f"  (목록: {jsons_missing_image})")

    # --- 2 & 3. 흠집 및 크기 분포 검사 (매핑이 일치하는 파일 기준) ---
    print("\n[흠집 및 차량 크기 분포 검사 (매핑 일치 파일 기준)]")
    matching_files = list(image_files.intersection(json_files))
    
    # 통계판 초기화 (small, mid, large로 변경)
    damage_counts = {name: 0 for name in damage_map.keys()}
    size_counts = {"small": 0, "mid": 0, "large": 0, "알수없음": 0}
    normal_count = 0
    unknown_labels = set()
    
    for base_name in matching_files:
        json_path = os.path.join(json_dir, base_name + '.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # --- 차량 크기 검사 ---
            size_category = "알수없음"
            if 'categories' in data and isinstance(data['categories'], dict) and 'supercategory_name' in data['categories']:
                json_size_name = data['categories']['supercategory_name']
                size_category = get_size_category(json_size_name) # 수정된 함수 사용
            size_counts[size_category] += 1
            
            # --- 흠집 종류 검사 (파일 내 모든 흠집 집계) ---
            if 'annotations' in data and data['annotations']:
                found_damages_in_file = set()
                
                for ann in data['annotations']: # 리스트의 모든 항목(ann)을 순회
                    damage_str = ann.get('damage', 'Unknown')
                    
                    if damage_str == 'Scratched':
                        damage_str = 'Scratch'
                        
                    if damage_str in damage_map:
                        found_damages_in_file.add(damage_str)
                    else:
                        unknown_labels.add(damage_str)
                
                if not found_damages_in_file:
                     normal_count += 1
                for damage_type in found_damages_in_file:
                    damage_counts[damage_type] += 1
            else:
                normal_count += 1
        except Exception as e:
            print(f"[경고] {json_path} 파일 처리 중 오류 발생: {e}")

    print("--- 📊 4종 흠집 분포 결과 ---")
    print("(참고: 한 파일에 여러 종류의 흠집이 있으면 각각 1씩 카운트됩니다.)")
    for name, count in damage_counts.items():
        print(f"- {name}: {count}개")
    print("-" * 30)
    print(f"- (정상 / Annotations 없음): {normal_count}개")
    if unknown_labels:
        print(f"- [주의] 알 수 없는 라벨: {list(unknown_labels)}")
    
    print("\n--- 🚗 차량 크기 분포 결과 ---")
    for name, count in size_counts.items():
        print(f"- {name}: {count}개")
        
    print(f"\n진단 완료. (처리된 파일: {len(matching_files)}개)")
    print("-" * 40)

# --- 메인 실행 ---
if __name__ == "__main__":
    # TL 데이터셋 진단
    tl_image_dir = './data/images/TL_image'
    tl_json_dir = './data/jsons/TL_json'
    check_dataset_health(tl_image_dir, tl_json_dir, "TL (학습/검증용)")

    # EL 데이터셋 진단
    el_image_dir = './data/images/EL_image'
    el_json_dir = './data/jsons/EL_json'
    check_dataset_health(el_image_dir, el_json_dir, "EL (평가용)")