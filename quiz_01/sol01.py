# 광고 데이터를 이용한 판매량 예측 프로그램
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 로드
df = pd.read_csv('quiz_01/Advertising.csv', encoding='utf-8')

# 데이터 전처리 (결측치 제거)
df = df.drop(columns='Unnamed: 0', axis=1)

# 독립변수와 종속변수 설정
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 생성
model = LinearRegression()

# 모델 훈련
model.fit(X_train, y_train)

# 테스트 데이터 예측
model.predict(X_test)

# 모델평가: R-squared(결정계수) 성능 평가
y_pred = model.predict(X_test)

# 새로운 데이터 예측
new_data = pd.DataFrame({'TV': [200], 'Radio': [50], 'Newspaper': [30]})

# 결과 출력
print(f"테스트 데이터에 대한 R-squared: {r2_score(y_test, y_pred)}")
print(f"새로운 데이터에 대한 예측: {model.predict(new_data)}")

# 실제 판매량 vs 예측 판매량 산점도
# x축: 실제 판매량. y축: 예측 판매량
# 제목과 축 레이블을 명확하게 표시
# 회귀선을 추가하여 시각화
plt.figure(figsize=(10, 6))
plt.title("Actual vs Predicted Sales", fontsize=15)
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Predicted')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 각 독립변수와 판매량과의 관계 시각화: 각 독립변수와 Sales 간의 산점도
# 회귀선을 추가하여 시각화
features = ['TV', 'Radio', 'Newspaper']
colors = ['red', 'green', 'blue']
plt.figure(figsize=(18, 6))
for i, (feature, color) in enumerate(zip(features, colors)):
    X_feature = df[[feature]]
    model_feature = LinearRegression()
    model_feature.fit(X_feature, y)
    y_pred_feature = model_feature.predict(X_feature)  # 전체 데이터에 대한 예측

    # subplot
    plt.subplot(1, 3, i+1)
    plt.scatter(X_feature, y, color=color, alpha=0.6, label=f'{feature} data')
    plt.plot(X_feature, y_pred_feature, color='black', label=f'Regression Line')
    plt.xlabel(f'{feature} Advertising Budget')
    plt.ylabel('Sales')
    plt.title(f'{feature} vs Sales')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 실제 판매량 vs 예측 판매량 산점도 (seaborn)
# 원래의 y_test와 y_pred 사용 (길이가 맞음)
result_df = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})

plt.figure(figsize=(10, 6))
# 다양한 색상 옵션 예시
sns.scatterplot(data=result_df, x='Actual Sales', y='Predicted Sales', 
                alpha=0.6, color='blue', s=50, label='Data')
sns.lineplot(x='Actual Sales', y='Actual Sales', data=result_df, 
             color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title("Actual vs Predicted Sales", fontsize=15)
plt.xlabel("Actual Sales", fontsize=12)
plt.ylabel("Predicted Sales", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 각 독립변수와 판매량과의 관계 시각화: 각 독립변수와 Sales 간의 산점도 (seaborn)
# 회귀선을 추가하여 시각화
plt.figure(figsize=(18, 6))
for i, feature in enumerate(['TV', 'Radio', 'Newspaper']):
    plt.subplot(1, 3, i+1)
    sns.regplot(data=df, x=feature, y='Sales', scatter_kws={'alpha':0.6}, line_kws={'color': 'black'})
    plt.title(f'{feature} vs Sales')
    plt.xlabel(f'{feature} Advertising Budget')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()