import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import joblib
import pandas as pd
import warnings

# --- 0. 설정 ---
warnings.filterwarnings('ignore')

# --- 1. "AI 눈 (Eyes)" 모델 로딩 (결함 인식) ---
damage_map_eye = {"Scratch": 0, "Crushed": 1, "Breakage": 2, "Separated": 3}
type_map_eye = {v: k for k, v in damage_map_eye.items()}

# "AI 눈" 모델 구조 생성 (train.py와 동일해야 함)
model_eye = models.resnet50(weights=None)
num_ftrs = model_eye.fc.in_features

# --- ★★★ train.py와 동일하게 Dropout 구조로 수정 ★★★ ---
model_eye.fc = nn.Sequential(
    nn.Dropout(p=0.5), # train.py와 동일한 구조
    nn.Linear(num_ftrs, len(damage_map_eye))
)
# --- ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★ ---

# 학습된 "AI 눈" 가중치 불러오기
try:
    model_eye.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
except FileNotFoundError:
    print("[오류] 'AI 눈' 모델(best_model.pth)을 찾을 수 없습니다.")
    print("먼저 train.py를 실행하여 모델을 훈련시켜야 합니다.")
    exit()
except RuntimeError as e:
    print(f"[오류] 'AI 눈' 모델 가중치 불러오기 실패: {e}")
    print("app.py와 train.py의 모델 구조가 일치하는지 확인하세요.")
    exit()
model_eye.eval()

# "AI 눈"을 위한 이미지 변환
transform_eye = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("✅ 'AI 눈(Eyes)' 모델(best_model.pth) 로딩 완료.")

# --- 2. "AI 뇌 (Brain)" 모델 로딩 (견적 예측) ---
MODEL_BRAIN_PATH = r"C:\Users\jjuyu\OneDrive\바탕 화면\FLAWNET_Project\model_brain.pkl"
try:
    model_brain = joblib.load(MODEL_BRAIN_PATH)
except FileNotFoundError:
    print(f"[오류] 'AI 뇌' 모델({MODEL_BRAIN_PATH})을 찾을 수 없습니다.")
    exit()
print("✅ 'AI 뇌(Brain)' 모델(model_brain.pkl) 로딩 완료.")


# --- 3. ★★★ AI "눈"과 "뇌"를 연결하는 예측 함수 ★★★ ---
def predict_flawnet_pipeline(image):
    """
    1단계: "AI 눈"이 이미지로 결함 종류를 예측
    2단계: "AI 뇌"가 결함 종류에 기반해 견적을 예측
    """
    
    # --- 1단계: "AI 눈 (Eyes)" 작동 ---
    image_tensor = transform_eye(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_eye(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, pred_idx = torch.max(probabilities, 0)
        
        defect_name = type_map_eye[pred_idx.item()]
        defect_confidence = float(confidence)

    # --- 2단계: "AI 뇌 (Brain)" 작동 ---
    # (시나리오 기반 입력 데이터 생성 - 이전과 동일)
    if defect_name == "Scratch":
        brain_input = {
            'car_model': '아반떼(20)-CN7', 'mileage_km': 10589,
            'num_exchange': 0, 'num_repair': 1, 'num_sheet_metal': 0,
            'num_painting': 1, 'num_detach': 10
        }
    elif defect_name == "Crushed":
        brain_input = {
            'car_model': '셀토스(19)', 'mileage_km': 13211,
            'num_exchange': 0, 'num_repair': 0, 'num_sheet_metal': 2,
            'num_painting': 2, 'num_detach': 4
        }
    elif defect_name == "Breakage":
        brain_input = {
            'car_model': '아반떼AD(16)', 'mileage_km': 53466,
            'num_exchange': 1, 'num_repair': 0, 'num_sheet_metal': 0,
            'num_painting': 1, 'num_detach': 5
        }
    else: # "Separated"
        brain_input = {
            'car_model': '올뉴쏘렌토(18)', 'mileage_km': 101270,
            'num_exchange': 0, 'num_repair': 1, 'num_sheet_metal': 1,
            'num_painting': 1, 'num_detach': 5
        }

    new_data = pd.DataFrame([brain_input])
    predicted_cost = model_brain.predict(new_data)
    final_cost = int(predicted_cost[0])

    # --- 3단계: 최종 결과 반환 ---
    eye_result = f"{defect_name} (신뢰도: {defect_confidence*100:.1f}%)"
    brain_result = f"약 {final_cost:,} 원"
    
    return eye_result, brain_result

# --- 4. Gradio 인터페이스 생성 (이전과 동일) ---
iface = gr.Interface(
    fn=predict_flawnet_pipeline,
    inputs=gr.Image(type="pil", label="결함이 있는 차량 이미지를 업로드하세요."),
    outputs=[
        gr.Label(label="🤖 AI '눈(Eyes)' 분석 결과 (결함 유형)"),
        gr.Label(label="💰 AI '뇌(Brain)' 분석 결과 (예상 견적)")
    ],
    title="FLAWNET v2: 차량 결함 분석 및 견적 예측 AI",
    description="''AI 눈''이 결함을 분류하고, ''AI 뇌''가 12만개 데이터를 기반으로 견적을 예측합니다."
)

if __name__ == "__main__":
    iface.launch()