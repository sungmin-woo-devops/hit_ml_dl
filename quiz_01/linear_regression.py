# 광고 데이터를 이용한 판매량 예측 프로그램
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 한글 폰트 설정
import matplotlib.font_manager as fm

# Windows 환경에서 한글 폰트 설정
try:
    # 사용 가능한 폰트 목록 확인
    font_list = []
    
    # Windows 기본 한글 폰트들
    windows_fonts = ['Malgun Gothic', 'NanumGothic', 'Gulim', 'Dotum', 'Batang', 'Arial Unicode MS']
    
    # matplotlib에서 사용 가능한 폰트들 중 한글 폰트 찾기
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in windows_fonts:
        if font in available_fonts:
            font_list.append(font)
    
    # 추가로 한글이 포함된 폰트들 찾기
    for font in available_fonts:
        if any(korean in font for korean in ['Gothic', 'Gulim', 'Dotum', 'Batang', 'Nanum']):
            font_list.append(font)
    
    print(f"사용 가능한 한글 폰트들: {font_list}")
    
    font_found = False
    for font in font_list:
        try:
            plt.rcParams['font.family'] = font
            # 테스트용 텍스트로 폰트 확인
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '한글테스트', fontsize=12, ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.close(fig)
            font_found = True
            print(f"한글 폰트 설정 완료: {font}")
            break
        except Exception as e:
            print(f"폰트 {font} 테스트 실패: {e}")
            continue
    
    if not font_found:
        # 폰트를 찾지 못한 경우 기본 설정
        plt.rcParams['font.family'] = 'DejaVu Sans'
        print("한글 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")
        
except Exception as e:
    print(f"폰트 설정 중 오류: {e}")
    plt.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False

# seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")

project_path = Path(__file__).parent
print(project_path)

# 1. 데이터 로드
print("=== 1. 데이터 로드 ===")
df = pd.read_csv(project_path / "dataset/Advertising.csv")
print("데이터 형태:", df.shape)
print("데이터 미리보기:")
print(df.head())

# 2. 데이터 전처리
print("\n=== 2. 데이터 전처리 ===")
df = df.drop(columns='Unnamed: 0', axis=1)
print("불필요한 컬럼 제거 후 데이터 형태:", df.shape)

# 독립변수와 종속변수 분리
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
print("독립변수 (X):", X.columns.tolist())
print("종속변수 (y):", y.name)

# 3. 데이터 분할
print("\n=== 3. 데이터 분할 ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"학습 데이터: {X_train.shape[0]}개")
print(f"테스트 데이터: {X_test.shape[0]}개")

# 4. 모델 학습
print("\n=== 4. 모델 학습 ===")
model = LinearRegression()
model.fit(X_train, y_train)
print("선형 회귀 모델 학습 완료")

# 5. 예측
print("\n=== 5. 예측 ===")
y_pred = model.predict(X_test)
print("테스트 데이터 예측 완료")

# 6. 모델 평가
print("\n=== 6. 모델 평가 ===")
r2 = r2_score(y_test, y_pred)
print(f"R-squared (결정 계수): {r2:.4f}")

# 7. 새로운 데이터 예측
print("\n=== 7. 새로운 데이터 예측 ===")
new_data = np.array([[200, 50, 30]])  # TV=200, Radio=50, Newspaper=30
predicted_sales = model.predict(new_data)[0]
print(f"TV=200, Radio=50, Newspaper=30일 때 예측 판매량: {predicted_sales:.2f}")

# 8. 결과 출력
print("\n=== 8. 결과 출력 ===")
print(f"테스트 데이터 R-squared 값: {r2:.4f}")
print(f"새로운 데이터 예측 판매량: {predicted_sales:.2f}")

# 9. 시각화
print("\n=== 9. 시각화 ===")

# 9-1. 실제 판매량 vs 예측 판매량 산점도
plt.figure(figsize=(10, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, s=100)

# 완벽한 예측선 추가
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction Line')

plt.xlabel('Actual Sales', fontsize=14)
plt.ylabel('Predicted Sales', fontsize=14)
plt.title('Actual Sales vs Predicted Sales', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9-2. 각 독립변수와 판매량과의 관계 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# TV와 판매량
sns.regplot(data=df, x='TV', y='Sales', ax=axes[0], scatter_kws={'alpha':0.6})
axes[0].set_title('TV Advertising vs Sales', fontsize=14, fontweight='bold')
axes[0].set_xlabel('TV Advertising ($)', fontsize=12)
axes[0].set_ylabel('Sales', fontsize=12)

# Radio와 판매량
sns.regplot(data=df, x='Radio', y='Sales', ax=axes[1], scatter_kws={'alpha':0.6})
axes[1].set_title('Radio Advertising vs Sales', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Radio Advertising ($)', fontsize=12)
axes[1].set_ylabel('Sales', fontsize=12)

# Newspaper와 판매량
sns.regplot(data=df, x='Newspaper', y='Sales', ax=axes[2], scatter_kws={'alpha':0.6})
axes[2].set_title('Newspaper Advertising vs Sales', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Newspaper Advertising ($)', fontsize=12)
axes[2].set_ylabel('Sales', fontsize=12)

plt.tight_layout()
plt.show()

print("=== 모든 단계 완료 ===")
print("1. 데이터 로드 ✓")
print("2. 데이터 전처리 ✓")
print("3. 데이터 분할 ✓")
print("4. 모델 학습 ✓")
print("5. 예측 ✓")
print("6. 모델 평가 ✓")
print("7. 새로운 데이터 예측 ✓")
print("8. 결과 출력 ✓")
print("9. 시각화 ✓") 