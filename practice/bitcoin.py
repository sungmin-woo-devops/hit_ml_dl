import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import ccxt
import optuna

# 시각화 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Bithumb API로 BTC/KRW, ETH/KRW 데이터 가져오기
def get_bithumb_data(market, start_date, end_date, count=215):
    exchange = ccxt.bithumb()
    symbol = f"{market}/KRW"
    timeframe = '1d'
    since = int(pd.to_datetime(start_date).timestamp() * 1000)
    candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=count)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['close']].rename(columns={'close': f"{market}_KRW"})
    return df.loc[start_date:end_date]

# CoinGecko API로 ETH/BTC 데이터 가져오기
def get_coingecko_data(coin, vs_currency, start_date, end_date):
    start_ts = int(pd.to_datetime(start_date).timestamp())
    end_ts = int(pd.to_datetime(end_date).timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart/range"
    params = {
        'vs_currency': vs_currency,
        'from': start_ts,
        'to': end_ts
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # HTTP 오류 체크
        data = response.json()
        
        # API 응답 구조 확인
        if 'prices' not in data:
            print(f"CoinGecko API 응답에 'prices' 키가 없습니다. 응답: {data}")
            return None
            
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        prices.set_index('timestamp', inplace=True)
        return prices['price'].rename(name=f"{coin.upper()}_{vs_currency.upper()}")
    except Exception as e:
        print(f"CoinGecko API 호출 중 오류: {e}")
        return None

# BTC Dominance (합성 데이터, CoinGecko 제한으로)
def get_btc_dominance(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dominance = 60.86 + np.random.normal(0, 0.5, len(dates))  # 60.86% 기준
    return pd.Series(dominance, index=dates, name='BTC_Dominance')

# Yahoo Finance로 WTI, USD/KRW, GOLD 데이터 가져오기
def get_yfinance_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        return df['Close']
    except Exception as e:
        print(f"Yahoo Finance {ticker} 데이터 수집 오류: {e}")
        return None

# 데이터 수집
start_date = '2024-01-01'
end_date = '2024-07-31'
actual_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Bithumb 데이터 (합성 데이터로 대체)
print("합성 데이터 생성 중...")
np.random.seed(42)  # 재현성을 위한 시드 설정

# 현실적인 암호화폐 가격 시뮬레이션
btc_base = 1.62e8  # 1억 6천만원 기준
eth_base = 5.28e6  # 528만원 기준

# 시간에 따른 트렌드와 변동성 추가
time_trend = np.linspace(0, 1, len(actual_dates))
btc_trend = 1 + 0.3 * np.sin(2 * np.pi * time_trend) + 0.1 * time_trend
eth_trend = 1 + 0.4 * np.sin(2 * np.pi * time_trend * 1.2) + 0.15 * time_trend

btc_krw = pd.Series(
    btc_base * btc_trend + np.cumsum(np.random.normal(0, 2e6, len(actual_dates))),
    index=actual_dates, name='BTC_KRW'
)
eth_krw = pd.Series(
    eth_base * eth_trend + np.cumsum(np.random.normal(0, 8e4, len(actual_dates))),
    index=actual_dates, name='ETH_KRW'
)

# Yahoo Finance 데이터 (합성 데이터로 대체)
wti_base = 70
usd_krw_base = 1300
gold_base = 2050

wti = pd.Series(
    wti_base + 10 * np.sin(2 * np.pi * time_trend * 0.5) + np.cumsum(np.random.normal(0, 2, len(actual_dates))),
    index=actual_dates, name='WTI'
)
usd_krw = pd.Series(
    usd_krw_base + 50 * np.sin(2 * np.pi * time_trend * 0.3) + np.cumsum(np.random.normal(0, 5, len(actual_dates))),
    index=actual_dates, name='USD_KRW'
)
gold = pd.Series(
    gold_base + 100 * np.sin(2 * np.pi * time_trend * 0.7) + np.cumsum(np.random.normal(0, 10, len(actual_dates))),
    index=actual_dates, name='GOLD'
)

# ETH/BTC 비율 (현실적인 범위)
eth_btc = pd.Series(
    0.0323 + 0.005 * np.sin(2 * np.pi * time_trend * 0.8) + np.random.normal(0, 0.0002, len(actual_dates)),
    index=actual_dates, name='ETH_BTC'
)

# BTC Dominance (현실적인 범위)
btc_dominance = pd.Series(
    60.86 + 5 * np.sin(2 * np.pi * time_trend * 0.4) + np.random.normal(0, 0.5, len(actual_dates)),
    index=actual_dates, name='BTC_Dominance'
)

# 데이터프레임 병합
data = pd.concat([btc_krw, eth_krw, eth_btc, wti, usd_krw, gold, btc_dominance], axis=1)
data.columns = ['BTC_KRW', 'ETH_KRW', 'ETH_BTC', 'WTI', 'USD_KRW', 'GOLD', 'BTC_Dominance']

# 특징 및 타겟 설정 - 실험 재설계
# 타겟: BTC/KRW, ETH/KRW
# 특징: USD/KRW, WTI, GOLD, ETH/BTC, BTC_Dominance + 시계열 특징

# 시계열 특징 추가
data['BTC_KRW_lag1'] = data['BTC_KRW'].shift(1)
data['ETH_KRW_lag1'] = data['ETH_KRW'].shift(1)
data['BTC_KRW_ma7'] = data['BTC_KRW'].rolling(window=7).mean()
data['ETH_KRW_ma7'] = data['ETH_KRW'].rolling(window=7).mean()
data['BTC_KRW_volatility'] = data['BTC_KRW'].rolling(window=7).std()
data['ETH_KRW_volatility'] = data['ETH_KRW'].rolling(window=7).std()

# USD/KRW와 GOLD의 상관관계 확인을 위한 추가 특징
data['USD_GOLD_ratio'] = data['USD_KRW'] / data['GOLD']
data['WTI_GOLD_ratio'] = data['WTI'] / data['GOLD']

# 결측치 처리
data = data.ffill().dropna()

print(f"데이터 형태: {data.shape}")
print(f"결측치 확인:\n{data.isnull().sum()}")

# 데이터 검증
if data.empty:
    raise ValueError("데이터가 비어있습니다. 데이터 생성 과정을 확인해주세요.")

if data.isnull().any().any():
    print("경고: 여전히 결측치가 있습니다.")
    data = data.dropna()
    print(f"결측치 제거 후 데이터 형태: {data.shape}")

# 특징 변수 재설정
feature_columns = [
    'USD_KRW', 'WTI', 'GOLD', 'ETH_BTC', 'BTC_Dominance',
    'BTC_KRW_lag1', 'ETH_KRW_lag1', 
    'BTC_KRW_ma7', 'ETH_KRW_ma7',
    'BTC_KRW_volatility', 'ETH_KRW_volatility',
    'USD_GOLD_ratio', 'WTI_GOLD_ratio'
]

X = data[feature_columns]
y_btc = data['BTC_KRW']
y_eth = data['ETH_KRW']

print("=== 특징 변수 상관관계 분석 ===")
print(f"X 데이터 형태: {X.shape}")
print(f"y_btc 데이터 형태: {y_btc.shape}")

# 데이터 유효성 검사
if X.isnull().any().any():
    print("경고: X 데이터에 결측치가 있습니다.")
    X = X.dropna()
    y_btc = y_btc[X.index]
    y_eth = y_eth[X.index]

correlation_with_target = X.corrwith(y_btc).sort_values(ascending=False)
print("BTC/KRW와의 상관관계:")
for feature, corr in correlation_with_target.items():
    print(f"  {feature}: {corr:.4f}")

print("\n=== USD/KRW와 GOLD 상관관계 ===")
usd_gold_corr = data['USD_KRW'].corr(data['GOLD'])
print(f"USD/KRW vs GOLD 상관계수: {usd_gold_corr:.4f}")

# 데이터 정규화 (로그 변환)
X_log = X.copy()
y_btc_log = np.log(y_btc)
y_eth_log = np.log(y_eth)

print(f"로그 변환 후 데이터 형태:")
print(f"X_log: {X_log.shape}")
print(f"y_btc_log: {y_btc_log.shape}")
print(f"y_eth_log: {y_eth_log.shape}")

# 훈련/테스트 분할 (시간 순서 유지)
split_idx = int(len(data) * 0.8)
X_train = X_log[:split_idx]
X_test = X_log[split_idx:]
y_btc_train = y_btc_log[:split_idx]
y_btc_test = y_btc_log[split_idx:]
y_eth_train = y_eth_log[:split_idx]
y_eth_test = y_eth_log[split_idx:]

# 특징 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델별 하이퍼파라미터 탐색
def objective_poly(trial):
    degree = trial.suggest_int('degree', 1, 3)
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train_scaled, y_btc_train)
    y_pred = model.predict(X_test_scaled)
    return mean_squared_error(y_btc_test, y_pred)

def objective_rf(trial):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
        max_depth=trial.suggest_int('max_depth', 3, 15),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
        random_state=42
    )
    model.fit(X_train_scaled, y_btc_train)
    y_pred = model.predict(X_test_scaled)
    return mean_squared_error(y_btc_test, y_pred)

def objective_xgb(trial):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 7),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        random_state=42
    )
    model.fit(X_train_scaled, y_btc_train)
    y_pred = model.predict(X_test_scaled)
    return mean_squared_error(y_btc_test, y_pred)

# 각 모델별 하이퍼파라미터 최적화
print("=== Polynomial Regression 하이퍼파라미터 최적화 ===")
study_poly = optuna.create_study(direction='minimize')
study_poly.optimize(objective_poly, n_trials=20)
print(f"최적 하이퍼파라미터: {study_poly.best_params}")
print(f"최적 MSE: {study_poly.best_value:.2e}")

print("\n=== Random Forest 하이퍼파라미터 최적화 ===")
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=30)
print(f"최적 하이퍼파라미터: {study_rf.best_params}")
print(f"최적 MSE: {study_rf.best_value:.2e}")

print("\n=== XGBoost 하이퍼파라미터 최적화 ===")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=30)
print(f"최적 하이퍼파라미터: {study_xgb.best_params}")
print(f"최적 MSE: {study_xgb.best_value:.2e}")


# 최적화된 하이퍼파라미터로 모델 초기화
polyreg = make_pipeline(PolynomialFeatures(degree=study_poly.best_params['degree']), LinearRegression())
rf_model = RandomForestRegressor(
    n_estimators=study_rf.best_params['n_estimators'],
    max_depth=study_rf.best_params['max_depth'],
    min_samples_split=study_rf.best_params['min_samples_split'],
    min_samples_leaf=study_rf.best_params['min_samples_leaf'],
    random_state=42
)
xgb_model = XGBRegressor(
    n_estimators=study_xgb.best_params['n_estimators'],
    learning_rate=study_xgb.best_params['learning_rate'],
    max_depth=study_xgb.best_params['max_depth'],
    min_child_weight=study_xgb.best_params['min_child_weight'],
    subsample=study_xgb.best_params['subsample'],
    colsample_bytree=study_xgb.best_params['colsample_bytree'],
    random_state=42
)

# 모델 평가 함수
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target_name):
    model.fit(X_train, y_train)
    y_pred_log = model.predict(X_test)
    
    # 로그 변환된 예측값을 원래 스케일로 변환
    y_pred = np.exp(y_pred_log)
    y_test_original = np.exp(y_test)
    
    mse = mean_squared_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    print(f"{model_name} ({target_name}) - MSE: {mse:.2e}, R2: {r2:.4f}")
    return y_pred

# BTC/KRW 예측
print("BTC/KRW 예측 결과:")
y_pred_btc_poly = evaluate_model(polyreg, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "Polynomial Regression", "BTC/KRW")
y_pred_btc_rf = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "Random Forest", "BTC/KRW")
y_pred_btc_xgb = evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "XGBoost", "BTC/KRW")

# ETH/KRW 예측
print("\nETH/KRW 예측 결과:")
y_pred_eth_poly = evaluate_model(polyreg, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "Polynomial Regression", "ETH/KRW")
y_pred_eth_rf = evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "Random Forest", "ETH/KRW")
y_pred_eth_xgb = evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "XGBoost", "ETH/KRW")

# 시각화 - 개별 차트로 생성

# 1. BTC/KRW 예측 결과
plt.figure(figsize=(12, 8))
sns.lineplot(data=pd.DataFrame({
    'Actual': np.exp(y_btc_test),
    'Polynomial': y_pred_btc_poly,
    'Random Forest': y_pred_btc_rf,
    'XGBoost': y_pred_btc_xgb
}), palette=['black', 'red', 'blue'], linewidth=2)
plt.title('BTC/KRW 예측 비교 (로그 변환 후)', fontsize=14, fontweight='bold')
plt.xlabel('테스트 샘플 인덱스', fontsize=12)
plt.ylabel('BTC/KRW 가격 (KRW)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. ETH/KRW 예측 결과
plt.figure(figsize=(12, 8))
sns.lineplot(data=pd.DataFrame({
    'Actual': np.exp(y_eth_test),
    'Polynomial': y_pred_eth_poly,
    'Random Forest': y_pred_eth_rf,
    'XGBoost': y_pred_eth_xgb
}), palette=['black', 'red', 'blue'], linewidth=2)
plt.title('ETH/KRW 예측 비교 (로그 변환 후)', fontsize=14, fontweight='bold')
plt.xlabel('테스트 샘플 인덱스', fontsize=12)
plt.ylabel('ETH/KRW 가격 (KRW)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. 하이퍼파라미터 최적화 결과 시각화
plt.figure(figsize=(10, 8))
model_names = ['Polynomial', 'Random Forest', 'XGBoost']
best_mse_values = [study_poly.best_value, study_rf.best_value, study_xgb.best_value]
colors = sns.color_palette("husl", 3)

bars = sns.barplot(x=model_names, y=best_mse_values, palette=colors, alpha=0.7)
plt.title('최적화된 모델별 MSE 비교', fontsize=14, fontweight='bold')
plt.ylabel('Mean Squared Error (log scale)', fontsize=12)
plt.yscale('log')
for i, (bar, mse) in enumerate(zip(bars.containers[0], best_mse_values)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1, 
             f'{mse:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# 4. R² 점수 비교
plt.figure(figsize=(10, 8))
btc_r2_scores = [
    r2_score(np.exp(y_btc_test), y_pred_btc_poly),
    r2_score(np.exp(y_btc_test), y_pred_btc_rf),
    r2_score(np.exp(y_btc_test), y_pred_btc_xgb)
]
eth_r2_scores = [
    r2_score(np.exp(y_eth_test), y_pred_eth_poly),
    r2_score(np.exp(y_eth_test), y_pred_eth_rf),
    r2_score(np.exp(y_eth_test), y_pred_eth_xgb)
]

r2_df = pd.DataFrame({
    'Model': model_names * 2,
    'R² Score': btc_r2_scores + eth_r2_scores,
    'Target': ['BTC/KRW'] * 3 + ['ETH/KRW'] * 3
})

sns.barplot(data=r2_df, x='Model', y='R² Score', hue='Target', palette=['skyblue', 'lightcoral'])
plt.title('모델별 R² 점수 비교 (로그 변환 후)', fontsize=14, fontweight='bold')
plt.ylabel('R² Score', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. 하이퍼파라미터 상세 정보
plt.figure(figsize=(12, 8))
plt.axis('off')

# 추가 시각화: 개선된 데이터 분석

# 10. 시계열 특징 변수 분포
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
sns.histplot(data=data, x='BTC_KRW_lag1', bins=30, kde=True, color='skyblue')
plt.title('BTC/KRW Lag1 분포', fontsize=12, fontweight='bold')

plt.subplot(2, 3, 2)
sns.histplot(data=data, x='BTC_KRW_ma7', bins=30, kde=True, color='lightgreen')
plt.title('BTC/KRW 7일 이동평균', fontsize=12, fontweight='bold')

plt.subplot(2, 3, 3)
sns.histplot(data=data, x='BTC_KRW_volatility', bins=30, kde=True, color='lightcoral')
plt.title('BTC/KRW 변동성', fontsize=12, fontweight='bold')

plt.subplot(2, 3, 4)
sns.histplot(data=data, x='USD_GOLD_ratio', bins=30, kde=True, color='gold')
plt.title('USD/GOLD 비율', fontsize=12, fontweight='bold')

plt.subplot(2, 3, 5)
sns.histplot(data=data, x='WTI_GOLD_ratio', bins=30, kde=True, color='orange')
plt.title('WTI/GOLD 비율', fontsize=12, fontweight='bold')

plt.subplot(2, 3, 6)
sns.scatterplot(data=data, x='USD_KRW', y='GOLD', alpha=0.6, color='purple')
plt.title('USD/KRW vs GOLD', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# 11. 개선된 상관관계 히트맵
plt.figure(figsize=(12, 10))
improved_correlation_matrix = data[feature_columns + ['BTC_KRW', 'ETH_KRW']].corr()
sns.heatmap(improved_correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('개선된 변수 간 상관관계', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 12. 시계열 특징의 예측력 분석
plt.figure(figsize=(15, 8))

plt.subplot(2, 2, 1)
plt.plot(data.index, data['BTC_KRW'], label='실제 BTC/KRW', alpha=0.7)
plt.plot(data.index, data['BTC_KRW_lag1'], label='BTC/KRW Lag1', alpha=0.7)
plt.title('BTC/KRW vs Lag1 비교', fontsize=12, fontweight='bold')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
plt.plot(data.index, data['BTC_KRW'], label='실제 BTC/KRW', alpha=0.7)
plt.plot(data.index, data['BTC_KRW_ma7'], label='BTC/KRW 7일 MA', alpha=0.7)
plt.title('BTC/KRW vs 7일 이동평균', fontsize=12, fontweight='bold')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plt.plot(data.index, data['USD_KRW'], label='USD/KRW', alpha=0.7)
plt.plot(data.index, data['GOLD'], label='GOLD', alpha=0.7)
plt.title('USD/KRW vs GOLD 시계열', fontsize=12, fontweight='bold')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
plt.plot(data.index, data['USD_GOLD_ratio'], label='USD/GOLD 비율', color='red', alpha=0.7)
plt.title('USD/GOLD 비율 시계열', fontsize=12, fontweight='bold')
plt.legend()
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 6. WTI 가격 분포
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='WTI', bins=30, kde=True, color='skyblue')
plt.title('WTI 가격 분포', fontsize=14, fontweight='bold')
plt.xlabel('WTI 가격 (USD)', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.tight_layout()
plt.show()

# 7. USD/KRW 환율 분포
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='USD_KRW', bins=30, kde=True, color='lightgreen')
plt.title('USD/KRW 환율 분포', fontsize=14, fontweight='bold')
plt.xlabel('USD/KRW 환율', fontsize=12)
plt.ylabel('빈도', fontsize=12)
plt.tight_layout()
plt.show()

# 8. 상관관계 히트맵
plt.figure(figsize=(10, 8))
correlation_matrix = data[['BTC_KRW', 'ETH_KRW', 'WTI', 'USD_KRW', 'GOLD', 'ETH_BTC', 'BTC_Dominance']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('변수 간 상관관계', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 9. BTC vs ETH 가격 관계
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='BTC_KRW', y='ETH_KRW', alpha=0.6, color='purple')
plt.title('BTC vs ETH 가격 관계', fontsize=14, fontweight='bold')
plt.xlabel('BTC/KRW 가격', fontsize=12)
plt.ylabel('ETH/KRW 가격', fontsize=12)
plt.tight_layout()
plt.show()