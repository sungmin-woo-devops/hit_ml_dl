# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 부호 깨짐 지정
plt.rcParams['axes.unicode_minus'] = False

# 숫자가 지수표현식으로 나올 때 지정
pd.options.display.float_format = '{:.2f}'.format

# 데이터 로드
df = pd.read_csv("./dataset/healthcare-dataset-stroke-data.csv", encoding='utf-8')
print("데이터 정보:")
print(df.info())
print("\n처음 5행:")
print(df.head())
print("\n컬럼명:")
print(df.columns)

# 결측치 확인 및 처리
print("\n결측치 확인:")
print(df.isnull().sum())
df['bmi'].fillna(df['bmi'].median(), inplace=True) # median 선택 이유: 이상치 존재
df = df[df['gender'] != 'Other'].drop(columns='id')

# 컬럼 타입 재분류 (id 제거 후)
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

# stroke는 타겟 변수이므로 특성에서 제외
numeric_features.remove('stroke')


print("\nUpdated Numeric columns: ", numeric_features)
print("Updated Categorical columns: ", categorical_features)

# 데이터 분할
X = df.drop("stroke", axis=1)
y = df["stroke"]

# 기초 통계 확인 (숫자형)
print("\n기초 통계:")
print(df.describe())

# 범주형 데이터 분포 확인
print("\n범주형 데이터 분포:")
for col in categorical_features:
    print(f"{col} 분포:")
    print(df[col].value_counts())
    print()

# 타겟 변수 불균형 확인
print("타겟 변수 분포:")
print(df['stroke'].value_counts(normalize=True))

# 전처리 구성
categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="passthrough"  # binary_features 그대로 사용
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

# 예측 및 평가
y_pred = model.predict(X_test_preprocessed)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n분류 보고서:")
print(report)
print("\n혼동 행렬:")
print(conf_matrix)

# 특성 중요도 확인
# 범주형 특성의 원-핫 인코딩된 컬럼명 생성
categorical_feature_names = []
for i, col in enumerate(categorical_features):
    categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i]
    categorical_feature_names.extend([f"{col}_{cat}" for cat in categories])

# 모든 특성명 결합
all_feature_names = numeric_features + categorical_feature_names

feature_importance = pd.DataFrame({
    'feature': all_feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n상위 10개 특성 중요도:")
print(feature_importance.head(10))

# 모델 성능 요약
accuracy = (968 + 2) / (968 + 4 + 48 + 2)
precision = 2 / (2 + 4)
recall = 2 / (2 + 48)
f1_score = 2 * precision * recall / (precision + recall)

print(f"\n모델 성능 요약:")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall): {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")

# 시각화 설정
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('뇌졸중 예측 모델 분석 결과', fontsize=16, fontweight='bold')

# 1. 혼동 행렬 시각화
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['뇌졸중 없음', '뇌졸중 있음'],
            yticklabels=['뇌졸중 없음', '뇌졸중 있음'],
            ax=axes[0,0])
axes[0,0].set_title('혼동 행렬 (Confusion Matrix)', fontweight='bold')
axes[0,0].set_xlabel('예측값')
axes[0,0].set_ylabel('실제값')

# 2. 모델 성능 지표 시각화
metrics = ['정확도', '정밀도', '재현율', 'F1-Score']
values = [accuracy, precision, recall, f1_score]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

bars = axes[0,1].bar(metrics, values, color=colors, alpha=0.7)
axes[0,1].set_title('모델 성능 지표', fontweight='bold')
axes[0,1].set_ylabel('점수')
axes[0,1].set_ylim(0, 1)

# 값 표시
for bar, value in zip(bars, values):
    height = bar.get_height()
    axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. 상위 10개 특성 중요도 시각화
top_features = feature_importance.head(10)
colors_importance = plt.cm.viridis(np.linspace(0, 1, len(top_features)))

bars = axes[1,0].barh(range(len(top_features)), top_features['importance'], 
                      color=colors_importance, alpha=0.7)
axes[1,0].set_yticks(range(len(top_features)))
axes[1,0].set_yticklabels(top_features['feature'], fontsize=9)
axes[1,0].set_title('상위 10개 특성 중요도', fontweight='bold')
axes[1,0].set_xlabel('중요도')

# 값 표시
for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
    axes[1,0].text(value + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=8)

# 4. 타겟 변수 분포 시각화
stroke_counts = df['stroke'].value_counts()
colors_pie = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = axes[1,1].pie(stroke_counts.values, 
                                          labels=['뇌졸중 없음', '뇌졸중 있음'],
                                          autopct='%1.1f%%', 
                                          colors=colors_pie,
                                          startangle=90)
axes[1,1].set_title('타겟 변수 분포', fontweight='bold')

# 텍스트 스타일 설정
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.show()

# 추가 시각화: 연령대별 뇌졸중 발생률
plt.figure(figsize=(12, 6))

# 연령대 구분
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 20, 40, 60, 80, 100], 
                         labels=['0-20', '21-40', '41-60', '61-80', '80+'])

age_stroke_rate = df.groupby('age_group')['stroke'].mean().sort_index()

plt.subplot(1, 2, 1)
bars = plt.bar(age_stroke_rate.index, age_stroke_rate.values, 
               color=plt.cm.Reds(np.linspace(0.3, 0.8, len(age_stroke_rate))))
plt.title('연령대별 뇌졸중 발생률', fontweight='bold', fontsize=12)
plt.xlabel('연령대')
plt.ylabel('뇌졸중 발생률')
plt.xticks(rotation=45)

# 값 표시
for bar, value in zip(bars, age_stroke_rate.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# 성별 뇌졸중 발생률
plt.subplot(1, 2, 2)
gender_stroke_rate = df.groupby('gender')['stroke'].mean()
colors_gender = ['#FF9999', '#66B2FF']
bars = plt.bar(gender_stroke_rate.index, gender_stroke_rate.values, color=colors_gender)
plt.title('성별 뇌졸중 발생률', fontweight='bold', fontsize=12)
plt.xlabel('성별')
plt.ylabel('뇌졸중 발생률')

# 값 표시
for bar, value in zip(bars, gender_stroke_rate.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📊 시각화 완료!")
print("• 혼동 행렬: 모델의 예측 성능을 한눈에 확인")
print("• 성능 지표: 정확도, 정밀도, 재현율, F1-Score 비교")
print("• 특성 중요도: 뇌졸중 예측에 가장 중요한 특성들")
print("• 타겟 분포: 데이터 불균형 현황")
print("• 연령대/성별 분석: 인구통계학적 특성별 뇌졸중 발생률") 