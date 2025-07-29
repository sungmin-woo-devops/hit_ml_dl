# Santander Bank 거래 예측 경진대회
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

print(os.getcwd())
print(os.listdir())

print("데이터 로드 중...")
# 데이터 로드
train_data = pd.read_csv("./practice/data/santander/train.csv", encoding='utf-8')
test_data = pd.read_csv("./practice/data/santander/test.csv", encoding='utf-8')
print("데이터 로드 완료!")

print("\n데이터 정보:")
print(train_data.info())
print(test_data.info())

print("\n결측값 확인:")
print(train_data.isnull().sum().sum(), "개의 결측값이 있습니다.")

# 피처와 타겟 분리
X = train_data.drop(columns=['ID_code', 'target'], axis=1)
y = train_data['target']
X_test = test_data.drop(columns=['ID_code'], axis=1)

print(f"\n원본 피처 수: {X.shape[1]}")

# 피처 엔지니어링: 행별 고유값 수 계산 (합성 데이터 식별에 도움)
X['unique_count'] = X.nunique(axis=1)
X_test['unique_count'] = X_test.nunique(axis=1)

print(f"피처 엔지니어링 후 피처 수: {X.shape[1]}")

# 피처 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 데이터 분할 (훈련/검증)
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n훈련 데이터 크기: {X_train.shape}")
print(f"검증 데이터 크기: {X_val.shape}")

# 모델 학습
print("\n모델 학습 중...")
model = GaussianNB()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC-AUC 점수: {roc_auc:.4f}")

# 피처 중요도 계산 (퍼뮤테이션 중요도)
print("\n피처 중요도 계산 중...")
perm_importance = permutation_importance(
    model, X_val, y_val, n_repeats=10, random_state=42, scoring='roc_auc'
)

# 시각화 설정
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Santander Bank 거래 예측 모델 분석', fontsize=16, fontweight='bold')

# 1. 상위 20개 피처 중요도 시각화
feature_names = [f'var_{i}' for i in range(200)] + ['unique_count']
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

top_20_features = feature_importance_df.head(20)

bars = axes[0,0].bar(range(len(top_20_features)), 
                     top_20_features['importance_mean'],
                     yerr=top_20_features['importance_std'],
                     capsize=5, alpha=0.7, color='skyblue')

axes[0,0].set_title('상위 20개 피처 중요도', fontweight='bold')
axes[0,0].set_xlabel('피처')
axes[0,0].set_ylabel('중요도')
axes[0,0].tick_params(axis='x', rotation=45)

# 값 표시
for i, (bar, mean_val) in enumerate(zip(bars, top_20_features['importance_mean'])):
    axes[0,0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                   f'{mean_val:.4f}', ha='center', va='bottom', fontsize=8)

# 2. ROC 곡선
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
axes[0,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', linewidth=2)
axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0,1].set_xlabel('False Positive Rate')
axes[0,1].set_ylabel('True Positive Rate')
axes[0,1].set_title('ROC 곡선', fontweight='bold')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. 타겟 변수 분포
target_counts = y.value_counts()
colors = ['#FF6B6B', '#4ECDC4']
wedges, texts, autotexts = axes[1,0].pie(target_counts.values, 
                                          labels=['거래 없음', '거래 있음'],
                                          autopct='%1.1f%%', 
                                          colors=colors,
                                          startangle=90)
axes[1,0].set_title('타겟 변수 분포', fontweight='bold')

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 4. 피처 중요도 히스토그램
axes[1,1].hist(perm_importance.importances_mean, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1,1].set_xlabel('중요도')
axes[1,1].set_ylabel('빈도')
axes[1,1].set_title('피처 중요도 분포', fontweight='bold')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('santander_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 추가 시각화: 상위 10개 피처 상세 분석
plt.figure(figsize=(12, 8))

top_10_features = feature_importance_df.head(10)
bars = plt.barh(range(len(top_10_features)), 
                top_10_features['importance_mean'],
                xerr=top_10_features['importance_std'],
                capsize=5, alpha=0.7, color=plt.cm.viridis(np.linspace(0, 1, len(top_10_features))))

plt.yticks(range(len(top_10_features)), top_10_features['feature'])
plt.xlabel('중요도')
plt.title('상위 10개 피처 중요도 (가로 막대)', fontweight='bold', fontsize=14)

# 값 표시
for i, (bar, mean_val, std_val) in enumerate(zip(bars, top_10_features['importance_mean'], top_10_features['importance_std'])):
    plt.text(bar.get_width() + std_val + 0.001, bar.get_y() + bar.get_height()/2,
             f'{mean_val:.4f} ± {std_val:.4f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('top_10_features.png', dpi=300, bbox_inches='tight')
plt.show()

# 모델 성능 요약
print("\n" + "="*50)
print("모델 성능 요약")
print("="*50)
print(f"ROC-AUC 점수: {roc_auc:.4f}")
print(f"훈련 데이터 크기: {X_train.shape[0]:,}")
print(f"검증 데이터 크기: {X_val.shape[0]:,}")
print(f"총 피처 수: {X.shape[1]}")
print(f"가장 중요한 피처: {top_10_features.iloc[0]['feature']} (중요도: {top_10_features.iloc[0]['importance_mean']:.4f})")

# 테스트 데이터 예측 및 제출 파일 생성
print("\n테스트 데이터 예측 중...")
test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
submission = pd.DataFrame({'ID_code': test_data['ID_code'], 'target': test_pred_proba})
submission.to_csv('santander_submission.csv', index=False)
print("제출 파일 'santander_submission.csv'가 생성되었습니다.")

print("\n📊 분석 완료!")
print("• 상위 20개 피처 중요도: 모델에 가장 영향을 주는 피처들")
print("• ROC 곡선: 모델의 분류 성능을 시각적으로 확인")
print("• 타겟 분포: 데이터 불균형 현황")
print("• 피처 중요도 분포: 전체 피처의 중요도 분포")
print("• 상위 10개 피처 상세 분석: 가장 중요한 피처들의 상세 정보") 