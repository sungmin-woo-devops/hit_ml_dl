import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def test_model_save_load():
    """모델 저장 및 로드 기능 테스트"""
    
    # 데이터 로드
    try:
        df = pd.read_csv("../dataset/healthcare-dataset-stroke-data.csv", encoding='utf-8')
        print("✅ 데이터 로드 성공")
    except FileNotFoundError:
        print("❌ 데이터 파일을 찾을 수 없습니다.")
        return
    
    # 데이터 전처리
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    df = df[df['gender'] != 'Other'].drop(columns='id')
    
    # 특성 분류
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features.remove('stroke')
    
    # 데이터 분할
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    # 전처리 구성
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )
    
    # 학습/데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    # 전처리 실행
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # SMOTE 적용
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
    
    # 모델 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # 모델 데이터 준비
    model_data = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': X.columns.tolist(),
        'best_params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 1}
    }
    
    # models 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 모델 저장
    joblib.dump(model_data, 'models/stroke_prediction_model.pkl')
    print("✅ 모델 저장 성공: models/stroke_prediction_model.pkl")
    
    # 모델 로드 테스트
    loaded_model_data = joblib.load('models/stroke_prediction_model.pkl')
    print("✅ 모델 로드 성공")
    
    # 예측 테스트
    test_input = pd.DataFrame({
        'age': [65],
        'gender': ['Male'],
        'hypertension': [1],
        'heart_disease': [0],
        'ever_married': ['Yes'],
        'work_type': ['Private'],
        'residence_type': ['Urban'],
        'avg_glucose_level': [120.0],
        'bmi': [28.0],
        'smoking_status': ['formerly smoked']
    })
    
    # 전처리 및 예측
    input_preprocessed = loaded_model_data['preprocessor'].transform(test_input)
    prediction = loaded_model_data['model'].predict(input_preprocessed)
    prediction_proba = loaded_model_data['model'].predict_proba(input_preprocessed)
    
    print(f"✅ 예측 테스트 성공:")
    print(f"   예측 결과: {'뇌졸중 위험' if prediction[0] == 1 else '정상'}")
    print(f"   뇌졸중 확률: {prediction_proba[0][1]:.2%}")
    
    # 성능 평가
    y_pred = loaded_model_data['model'].predict(X_test_preprocessed)
    report = classification_report(y_test, y_pred)
    print("\n📊 모델 성능:")
    print(report)

if __name__ == "__main__":
    test_model_save_load() 