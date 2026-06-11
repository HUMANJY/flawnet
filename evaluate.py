import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# train.py에서 FocalLoss와 가중치(weights_tensor)를 함께 가져옵니다.
from train import CarDamageDataset, damage_map, val_transform as test_transform, FocalLoss, weights_tensor
import datetime
import os
import time # --- ★★★ 'time' 모듈 import 추가 ★★★ ---

# --- 1. 한글 폰트 설정 ---
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # --- 2. 데이터 준비 ---
    EVAL_IMAGE_DIR = './data/images/EL_image'
    EVAL_JSON_DIR = './data/jsons/EL_json'
    
    try:
        eval_files = [f for f in os.listdir(EVAL_IMAGE_DIR) if f.endswith('.jpg')]
        print(f"EL 폴더({EVAL_IMAGE_DIR})에서 {len(eval_files)}개의 평가용 이미지 파일을 찾았습니다.")
    except FileNotFoundError:
        print(f"[오류] 평가 이미지 폴더 '{EVAL_IMAGE_DIR}'를 찾을 수 없습니다. 경로를 확인하세요.")
        exit()
        
    if not eval_files:
        print(f"[오류] 평가 이미지 폴더 '{EVAL_IMAGE_DIR}'에 이미지 파일이 없습니다.")
        exit()

    test_dataset = CarDamageDataset(EVAL_IMAGE_DIR, EVAL_JSON_DIR, eval_files, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 3. 모델 불러오기 ---
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    
    # --- train.py와 동일하게 Dropout 구조로 수정 ---
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5), # train.py와 동일한 구조
        nn.Linear(num_ftrs, len(damage_map))
    )
    # --- ★★★★★★★★★★★★★★★★★★★★★★★ ---

    try:
        model.load_state_dict(torch.load('best_model.pth'))
    except FileNotFoundError:
        print("[오류] 학습된 모델 파일 'best_model.pth'를 찾을 수 없습니다. 먼저 train.py를 실행하세요.")
        exit()
    except RuntimeError as e:
        print(f"[오류] 모델 가중치 불러오기 실패: {e}")
        print("train.py와 evaluate.py의 모델 구조가 일치하는지 다시 확인하세요.")
        exit()
        
    model.to(device)
    model.eval()
    
    # --- 평가용 손실 함수(Criterion) 정의 ---
    criterion = FocalLoss(alpha=weights_tensor, gamma=2.0).to(device)

    print("✅ 모델 및 평가 데이터 준비 완료. 평가를 시작합니다...")
    start_time = time.time() # --- 시간 측정 시작 ---

    # --- 4. 평가 실행 ---
    all_labels, all_preds = [], []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    if not all_labels:
        print(f"[오류] 평가 데이터셋({EVAL_IMAGE_DIR}, {EVAL_JSON_DIR})에서 라벨을 로드하지 못했습니다.")
        exit()

    # --- 5. 결과 계산 ---
    end_time = time.time() # --- 시간 측정 종료 ---
    duration = end_time - start_time
    
    avg_loss = total_loss / len(test_loader)
    overall_accuracy = accuracy_score(all_labels, all_preds)
    class_names = list(damage_map.keys())
    
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
    report_str = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    
    macro_f1 = report_dict['macro avg']['f1-score']
    cm = confusion_matrix(all_labels, all_preds)

    # --- 결과를 문자열로 만들기 ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"evaluation_results_{timestamp}.txt"
    
    summary_line = f"🎯 Loss: {avg_loss:.4f} | Accuracy: {overall_accuracy:.4f} | Macro F1: {macro_f1:.4f}"
    
    result_output = f"--- 모델 평가 결과 ({timestamp}) ---\n\n"
    result_output += f"⏱️ 평가 소요 시간: {duration:.2f} 초 ({duration/60:.2f} 분)\n\n"
    result_output += f"🎯 전체 정확도 (Overall Accuracy): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)\n"
    result_output += f"--- 📌 평가 요약 📌 ---\n{summary_line}\n\n"
    result_output += "--- 📊 결함 종류 분류 성능 ---\n"
    result_output += report_str
    result_output += "\n\n--- 혼동 행렬 (Confusion Matrix) ---\n"
    result_output += str(cm)

    # --- 6. 결과 파일 저장 ---
    with open(results_filename, 'w', encoding='utf-8') as f:
        f.write(result_output)
    
    print(f"\n✅ 평가 결과가 '{results_filename}' 파일에 저장되었습니다.")

    # --- 7. 동일한 결과 터미널 출력 ---
    print("\n" + result_output)

    # --- 8. 혼동 행렬 시각화 ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('예측 라벨')
    plt.ylabel('실제 라벨')
    plt.title('결함 종류 분류 혼동 행렬')
    plt.show()