# 광고 데이터를 이용한 판매량 예측 프로그램
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """데이터 로드 및 전처리"""
    # 데이터 로드
    df = pd.read_csv('quiz_01/Advertising.csv', encoding='utf-8')
    
    # 불필요한 컬럼 제거
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0', axis=1)
    
    return df

def train_model(X, y):
    """선형 회귀 모델 훈련"""
    # 훈련 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 모델 생성 및 훈련
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 예측
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

def evaluate_model(y_test, y_pred):
    """모델 성능 평가"""
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"=== 모델 성능 평가 ===")
    print(f"R-squared (결정계수): {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print()

def predict_new_data(model, new_data):
    """새로운 데이터에 대한 예측"""
    prediction = model.predict(new_data)
    print(f"=== 새로운 데이터 예측 ===")
    print(f"입력 데이터: {new_data.iloc[0].to_dict()}")
    print(f"예측 판매량: {prediction[0]:.2f}")
    print()

def plot_actual_vs_predicted(y_test, y_pred):
    """실제 vs 예측 판매량 시각화"""
    plt.figure(figsize=(10, 6))
    plt.title("실제 vs 예측 판매량", fontsize=15)
    plt.xlabel("실제 판매량", fontsize=12)
    plt.ylabel("예측 판매량", fontsize=12)
    
    plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='예측값')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='완벽한 예측선')
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_relationships(df):
    """각 독립변수와 판매량과의 관계 시각화"""
    features = ['TV', 'Radio', 'Newspaper']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('각 광고 매체와 판매량의 관계', fontsize=16)
    
    for i, feature in enumerate(features):
        # 산점도와 회귀선
        sns.regplot(data=df, x=feature, y='Sales', ax=axes[i], 
                    scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
        axes[i].set_title(f'{feature} vs Sales')
        axes[i].set_xlabel(f'{feature} 광고비')
        axes[i].set_ylabel('판매량')
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 함수"""
    print("=== 광고 데이터를 이용한 판매량 예측 프로그램 ===\n")
    
    # 데이터 로드 및 전처리
    df = load_and_preprocess_data()
    print(f"데이터 형태: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    print()
    
    # 독립변수와 종속변수 설정
    X = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    
    # 모델 훈련
    model, X_train, X_test, y_train, y_test, y_pred = train_model(X, y)
    
    # 모델 성능 평가
    evaluate_model(y_test, y_pred)
    
    # 새로운 데이터 예측
    new_data = pd.DataFrame({
        'TV': [200], 
        'Radio': [50], 
        'Newspaper': [30]
    })
    predict_new_data(model, new_data)
    
    # 시각화
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_relationships(df)
    
    # 모델 계수 출력
    print("=== 모델 계수 ===")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"{feature}: {coef:.4f}")
    print(f"절편: {model.intercept_:.4f}")

if __name__ == "__main__":
    main()