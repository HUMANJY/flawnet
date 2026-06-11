import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# 1. 1단계에서 만든 교과서(CSV) 불러오기
CSV_PATH = r"C:\Users\jjuyu\OneDrive\바탕 화면\FLAWNET_Project\data\brain_training_data.csv"
MODEL_SAVE_PATH = r"C:\Users\jjuyu\OneDrive\바탕 화면\FLAWNET_Project\model_brain.pkl"

try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"오류: {CSV_PATH} 파일을 찾을 수 없습니다.")
    print("먼저 1단계(1_data_preparation.py) 코드를 실행해주세요.")
    exit()

print(f"총 {len(df)}개의 데이터로 'AI 뇌(Brain)' 모델 학습을 시작합니다.")

# 2. X (문제지)와 y (정답지) 분리
X = df.drop('Y_TOTAL_COST_WON', axis=1)
y = df['Y_TOTAL_COST_WON']

# 3. AI가 학습할 수 있도록 데이터 전처리 설정
categorical_features = ['car_model']
numerical_features = ['mileage_km', 'num_exchange', 'num_repair', 'num_sheet_metal', 'num_painting', 'num_detach']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. 훈련용/테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 'Ridge' 모델을 포함한 전체 AI 파이프라인 정의
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# 6. 학습 시작!
print("AI 뇌(Brain) 모델 학습 중...")
pipeline.fit(X_train, y_train)

# 7. 학습 결과 평가
score = pipeline.score(X_test, y_test)
print(f"\n--- 학습 완료 ---")
print(f"모델 설명력 (R-squared): {score:.4f} (1.0에 가까울수록 좋습니다)")

# 8. 완성된 "AI 뇌(Brain)" 모델을 파일로 저장
joblib.dump(pipeline, MODEL_SAVE_PATH)
print(f"\n성공! 훈련된 'AI 뇌(Brain)' 모델을 '{MODEL_SAVE_PATH}' 파일로 저장했습니다.")