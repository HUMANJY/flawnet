import os
import json
from collections import defaultdict

# --- 1. 검사 기준 설정 ---
# train.py에서 사용하는 4종 흠집 맵
damage_map = {"Scratch": 0, "Crushed": 1, "Breakage": 2, "Separated": 3}
# data_diagnostics.py에서 사용하는 차량 크기 맵
size_map = {
    "소형": ["Compact car", "Subcompact car"],
    "중형": ["Mid-size car", "Crossover"],
    "대형": ["Full-size car", "SUV", "Truck", "Van", "Pickup truck"]
}

def get_size_category(json_size_name):
    """JSON의 supercategory_name을 소/중/대형으로 변환합니다."""
    for category, names in size_map.items():
        if json_size_name in names:
            return category
    return "알수없음"

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
        # .jpg 확장자를 제외한 파일 이름만 set으로 만듭니다.
        image_files = set(f.replace('.jpg', '') for f in os.listdir(image_dir) if f.endswith('.jpg'))
        # .json 확장자를 제외한 파일 이름만 set으로 만듭니다.
        json_files = set(f.replace('.json', '') for f in os.listdir(json_dir) if f.endswith('.json'))
    except FileNotFoundError as e:
        print(f"[오류] 폴더를 찾을 수 없습니다: {e.filename}")
        print("바탕화면에 'FLAWNET_Extracted_TL_image' 등의 폴더가 있는지 확인하세요.")
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
    
    # 통계판 초기화
    damage_counts = {name: 0 for name in damage_map.keys()}
    size_counts = {"소형": 0, "중형": 0, "대형": 0, "알수없음": 0}
    normal_count = 0
    unknown_labels = set()
    
    for base_name in matching_files:
        json_path = os.path.join(json_dir, base_name + '.json')
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # --- 차량 크기 검사 ---
                size_category = "알수없음"
                if 'categories' in data and 'supercategory_name' in data['categories']:
                    json_size_name = data['categories']['supercategory_name']
                    size_category = get_size_category(json_size_name)
                size_counts[size_category] += 1
                
                # --- 흠집 종류 검사 ---
                if 'annotations' in data and data['annotations']:
                    damage_str = data['annotations'][0].get('damage', 'Unknown')
                    
                    if damage_str == 'Scratched':
                        damage_str = 'Scratch'
                        
                    if damage_str in damage_counts:
                        damage_counts[damage_str] += 1
                    else:
                        unknown_labels.add(damage_str)
                else:
                    normal_count += 1
        except Exception as e:
            print(f"[경고] {json_path} 파일 처리 중 오류 발생: {e}")

    print("--- 📊 4종 흠집 분포 결과 ---")
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
    # 바탕화면 경로 설정
    DESKTOP_PATH = os.path.join(os.path.expanduser('~'), 'OneDrive', '바탕 화면')
    
    # TL 데이터셋 진단 (바탕화면의 새 폴더 경로)
    tl_image_dir = os.path.join(DESKTOP_PATH, 'FLAWNET_Extracted_TL_image')
    tl_json_dir = os.path.join(DESKTOP_PATH, 'FLAWNET_Extracted_TL_json')
    check_dataset_health(tl_image_dir, tl_json_dir, "TL (추출된 학습/검증용)")

    # EL 데이터셋 진단 (바탕화면의 새 폴더 경로)
    el_image_dir = os.path.join(DESKTOP_PATH, 'FLAWNET_Extracted_EL_image')
    el_json_dir = os.path.join(DESKTOP_PATH, 'FLAWNET_Extracted_EL_json')
    check_dataset_health(el_image_dir, el_json_dir, "EL (추출된 평가용)")