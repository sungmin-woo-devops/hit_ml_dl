#!/usr/bin/env python
# coding: utf-8

# # G-Research Crypto Forecasting 
# https://www.kaggle.com/competitions/g-research-crypto-forecasting/code
# 
# ### Reference
# - https://www.kaggle.com/competitions/g-research-crypto-forecasting/code
# - https://www.kaggle.com/code/lucasmorin/on-line-feature-engineering
# 
# ### 특성 공학
# 
# #### 암호화폐 데이터의 노이즈와 변동성을 고려할 때, 유의미한 특성을 생성하는 것이 핵심
# - 지연 특성(Lag Features): 이전 시간대의 수익률(예: t-1, t-2 시점의 값)
# - 이동 평균(Moving Averages): 5분, 15분 등 다양한 시간 단위의 평균 가격 또는 수익률
# - 변동성 지표(Volatility): 일정 구간 내 가격 변동의 표준편차
# - 기술적 지표: RSI, MACD와 같은 시장 분석 도구
# 
# ##### 지연 특성(Lag Features): 이전 시간대의 수익률(예: t-1, t-2 시점의 값)
# 과거 시점의 데이터를 활용하여 현재 데이터를 예측하는 데 사용.
# 예를 들어, 이전 시간대의 종가(Close) 또는 수익률을 특성으로 사용
# 중요 이유: 암호화폐 가격은 종종 자기 상관성(autocorrelation)을 가지므로, 과거 가격이 미래 가격에 영향을 줄 수 있음
# 예시: 분 단위 데이터에서 lag_1은 1분전 종가, lag_2는 2분전 종가를 의미합니다.
# 
# ##### 이동 평균(Moving Averages)
# - 일정 기간 동안의 가격 데이터를 평활화하여 장기적인 추세를 파악하는데 사용
# - 종류: SMA(Simple Moving Average), EMA(Exponential Moving Average)
#     - 단순 이동 평균: 최근 n 개 기간의 평균
#     - 지수 이동 평균: 최근 데이터에 더 많은 가중치를 두는 평균
# - 중요 이유: 암호화폐 시장의 변동성을 줄이고 추세를 명확히 파악하는데 유용
# - Close 가격 또는 수익률에 대해 5분, 15분 등의 시간 창(time window)으로 SMA와 EMA를 계산합니다.
# 
# ##### 변동성 지표(Volatility Indicators)
# - 변동성은 일정 기간 동안 가격 변동의 정도를 측정하며, 암호화폐의 높은 가격 변동성을 반영하는 데 필수적입니다.
# - 계산 방법: 수익률 또는 가격 변화의 표준편차를 일정 시간 창(예: 15분)으로 계산합니다.
# - 왜 중요한가?: 변동성은 거래 기회나 리스크를 나타내며, 모델이 시장의 불안정성을 이해하는 데 도움을 줍니다.
# - 구현 방법: 수익률의 15분 롤링 표준편차를 계산합니다.
# 
# ##### 기술적 지표(Technical Indicators)
# - RSI(상대강도지수), MACD(Moving Average Convergence Divergence)는 시장의 모멘텀과 추세를 파악하는데 사용
# - RSI: 가격 변화의 속도와 크기를 0~100 사이로 측정. RSI > 70은 과매수(overbought), RSI < 30은 과매도(oversold)를 나타냅니다.
# - MACD: 12기간과 26기간 EMA의 차이를 계산하고, 9기간 신호선(signal line)을 통해 추세 변화를 감지합니다.
# - 왜 중요한가?: 암호화폐의 급격한 가격 변동에서 모멘텀과 반전 시점을 포착하는 데 유용합니다.
# - 구현 방법: ta 라이브러리를 사용하거나 수동으로 RSI와 MACD를 계산합니다.
# 
# ##### 데이터 분할
# - 암호화폐 시장은 시간이 지남에 따라 진화하므로, 실시간 예측을 시뮬레이션하기 위해 시간 기반 데이터 분할이 필요
# - 시간 기반 분할 (Time-based split)
#     - 훈련 데이터: 초기 데이터(예: 70%)를 사용해 모델을 학습.
#     - 검증 데이터: 그 다음 연속된 시간 구간(예: 20%)를 사용해 하이퍼파라미터 튜닝.
#     - 테스트 데이터: 가장 최근 데이터(예: 10%)를 최종 평가에 사용.
# - 왜 중요한가?: 미래 데이터가 과거 예측에 영향을 주지 않도록 하여 데이터 누수(data leakage)를 방지
# - 구현방법: timestamp를 기준으로 데이터를 정렬하고 순차적으로 분할
# - 실시간 데이터 사용해 평가 -> 실험에서도 시간 기반 분할(Time-based split 적용)
# - 훈련 데이터: 과거 데이터의 초기 부분(예: 70%)
# - 검증 데이터: 이후 연속된 시간 구간(예: 20%)
# - 테스트 데이터: 마지막 구간(예: 10%) 또는 별도 준비된 미래 데이터
import os
print(os.getcwd())
print(os.listdir())
BASE_DIR = os.getcwd()
DATA_DIR = BASE_DIR + "/data/g-research-crypto-forecasting/"
print(DATA_DIR)
from datetime import datetime
import time
import pickle   # 모델 또는 데이터 직렬화
import gc   # 가비지 컬렉션으로 메모리 수동 관리
from tqdm import tqdm   # 프로그레스바
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
import optuna
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 6)
pd.set_option('display.max_columns', 350)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
# 대용량 CSV read 작업 중 메모리 오류 발생
# 로드 과정을 더 빠르게 할 수는 없을까?
# 훈련 데이터 로드
train_df = pd.read_csv(DATA_DIR + 'train.csv', encoding='utf-8') # nrows=400000
supplemental_df = pd.read_csv(DATA_DIR + 'supplemental_train.csv', encoding='utf-8')
test_df = pd.read_csv(DATA_DIR + 'example_test.csv', encoding='utf-8')
# 훈련 데이터 결합
train_df = pd.concat([train_df, supplemental_df], ignore_index=True)
# 중복 제거, 정렬 (인덱스 상태는 무관)
train_df = train_df.drop_duplicates().sort_values(by=['Asset_ID', 'timestamp'])
# 정렬 후 인덱스 초기화
train_df = train_df.reset_index(drop=True)
# 발생한 이슈
# Dask GroupBy 제한: GroupBy 객체는 Pandas와 달리 fillna를 직접 지원하지 않음
# fillna는 DataFrame 수준에서 적용되어야 함
# 결측치 제거
print(train_df.isna().sum().to_string()) # .isna == .isnull
print(train_df.columns)
train_df = train_df.drop(columns=['Target'])
# 정렬 전 인덱스 제거
train_df = train_df.reset_index(drop=True)
# 정렬
train_df = train_df.sort_values(['Asset_ID', 'timestamp'])
# groupby 후 선형 보간
train_df = train_df.groupby('Asset_ID').apply(lambda x: x.interpolate(method='linear'))
# 멀티 인덱스 제거
train_df.reset_index(drop=True, inplace=True)
# 결과 확인
print(train_df.head())
# 결측치 확인
missing_counts_after = train_df.isna().sum()
print("(after) 결측치 수:\n", missing_counts_after.to_string())
# 데이터 정렬
train_df = train_df.sort_values(by=['Asset_ID', 'timestamp'])
# 특성 공학 함수
def compute_features(df):
    """
    자산별로 특성 공학을 수행
    Args: df - pandas DataFrame
    Returns: 특성이 추가된 DataFrame
    """
    print(f"Before compute_features - Columns: {df.columns.tolist()}")
    if 'Close' not in df.columns:
        raise KeyError(f"Column 'Close' not found in DataFrame. Available columns: {df.columns.tolist()}")
    df = df.copy()
    # 1. 지연 특성 (Lag Features)
    df['lag_1_close'] = df['Close'].shift(1)
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    df['lag_1_returns'] = df['returns'].shift(1)
    # 2. 이동 평균 (Moving Averages)
    df['sma_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['ema_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    # 3. 변동성 지표 (Volatility Indicators)
    df['volatility_15'] = df['returns'].rolling(window=15, min_periods=1).std()
    df['high_low_spread'] = df['High'] - df['Low']
    bb = BollingerBands(close=df['Close'], window=20, window_dev=2, fillna=True)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    # 4. 기술적 지표 (Technical Indicators)
    rsi = RSIIndicator(close=df['Close'], window=14, fillna=True)
    df['rsi'] = rsi.rsi()
    macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    # 5. 가격 스프레드 (Price Spreads)
    df['close_open_spread'] = df['Close'] - df['Open']
    # 6. 거래량 기반 특성 (Volume-based Features)
    df['volume_log'] = np.log1p(df['Volume'])
    df['volume_roll_mean_5'] = df['Volume'].rolling(window=5, min_periods=1).mean()
    # 다중공선성 완화 (차등 학습)
    df['close_lag1_diff'] = df['Close'] - df['lag_1_close']
    df['close_sma5_ratio'] = df['Close'] / df['sma_5'] - 1
    print(f"After compute_features - Columns: {df.columns.tolist()}")
    return df
# 자산별 데이터 분리 및 특성 공학
asset_dfs = {}
for asset_id in train_df['Asset_ID'].unique():
    asset_df = train_df[train_df['Asset_ID'] == asset_id].copy()
    asset_df = compute_features(asset_df)
    asset_df = asset_df.dropna()  # 결측치 제거
    asset_dfs[asset_id] = asset_df
# 자산별 데이터 분할 (예: 70-20-10)
split_results = {}
for asset_id, df in asset_dfs.items():
    total_len = len(df)
    train_size = int(0.7 * total_len)
    val_size = int(0.2 * total_len)
    test_size = total_len - train_size - val_size
    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size + val_size]
    test_data = df.iloc[train_size + val_size:]
    # 스케일링 적용
    scaler_price = MinMaxScaler()
    scaler_volume = RobustScaler()
    scaler_target = StandardScaler()
    price_cols = ['Open', 'High', 'Low', 'Close', 'VWAP']
    volume_cols = ['Volume', 'Count']
    train_data[price_cols] = scaler_price.fit_transform(train_data[price_cols])
    train_data[volume_cols] = scaler_volume.fit_transform(np.log1p(train_data[volume_cols]))
    train_data['target'] = scaler_target.fit_transform(train_data[['target']])
    val_data[price_cols] = scaler_price.transform(val_data[price_cols])
    val_data[volume_cols] = scaler_volume.transform(np.log1p(val_data[volume_cols]))
    val_data['target'] = scaler_target.transform(val_data[['target']])
    test_data[price_cols] = scaler_price.transform(test_data[price_cols])
    test_data[volume_cols] = scaler_volume.transform(np.log1p(test_data[volume_cols]))
    test_data['target'] = scaler_target.transform(test_data[['target']])
    split_results[asset_id] = {'train': train_data, 'val': val_data, 'test': test_data}
    print(f"Asset {asset_id} - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
# 자산별 특성 공학 적용 (include_groups 제거)
train_df = train_df.groupby('Asset_ID', group_keys=False).apply(compute_features)
test_df = test_df.groupby('Asset_ID', group_keys=False).apply(compute_features)
# 결측치 처리
train_df = train_df.groupby('Asset_ID').apply(lambda x: x.interpolate(method='linear', limit_direction='both')).reset_index(drop=True)
test_df = test_df.groupby('Asset_ID').apply(lambda x: x.interpolate(method='linear', limit_direction='both')).reset_index(drop=True)
train_df = train_df.fillna(method='ffill').dropna()
test_df = test_df.fillna(method='ffill').dropna()
# 타겟 변수 생성 (훈련 데이터에만 적용)
train_df['target'] = train_df['returns'].shift(-1)
train_df = train_df.dropna()
# 다중공선성 제거 (Close 제외)
correlation_matrix = train_df.select_dtypes(include=[np.number]).corr().abs()
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85) and column != 'Close']
train_df = train_df.drop(columns=to_drop)
test_df = test_df.drop(columns=to_drop)
# 특성 선택
features = [col for col in train_df.columns if col not in ['timestamp', 'Asset_ID', 'Open', 'High', 'Low', 'Volume', 'VWAP', 'target']]
# 시간 기반 데이터 분할 (train_df에 적용)
train_df['datetime'] = pd.to_datetime(train_df['timestamp'], unit='s')
train_df = train_df.sort_values('timestamp')
total_len = len(train_df)
train_size = int(0.7 * total_len)
val_size = int(0.2 * total_len)
test_size = total_len - train_size - val_size
train_data = train_df.iloc[:train_size]
val_data = train_df.iloc[train_size:train_size + val_size]
test_data = train_df.iloc[train_size + val_size:]
# 스케일링 적용
scaler_price = MinMaxScaler()
scaler_volume = RobustScaler()
scaler_target = StandardScaler()
price_cols = ['Open', 'High', 'Low', 'Close', 'VWAP']
volume_cols = ['Volume', 'Count']
# 훈련 데이터 피팅 및 변환
train_data[price_cols] = scaler_price.fit_transform(train_data[price_cols])
train_data[volume_cols] = scaler_volume.fit_transform(np.log1p(train_data[volume_cols]))
train_data['target'] = scaler_target.fit_transform(train_data[['target']])
# 검증/테스트 데이터 변환
val_data[price_cols] = scaler_price.transform(val_data[price_cols])
val_data[volume_cols] = scaler_volume.transform(np.log1p(val_data[volume_cols]))
val_data['target'] = scaler_target.transform(val_data[['target']])
test_data[price_cols] = scaler_price.transform(test_data[price_cols])
test_data[volume_cols] = scaler_volume.transform(np.log1p(test_data[volume_cols]))
test_data['target'] = scaler_target.transform(test_data[['target']])
# 목적: 암호화폐 자산별 가격 상승여부 예측 → 분류(0/1) or 회귀 모두 가능
# 아래 예시는 회귀 (target: 수익률 등)
def objective(trial):
    param = {
        'tree_method': 'gpu_hist',          # GPU 연산 사용
        'predictor': 'gpu_predictor',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'random_state': 42,
        'objective': 'reg:squarederror',   # 회귀용. 분류일 경우 'binary:logistic'
        'verbosity': 0
    }
    model = xgb.XGBRegressor(**param)
    model.fit(
        train_data[features], train_data['target'],
        eval_set=[(val_data[features], val_data['target'])],
        early_stopping_rounds=10,
        verbose=False
    )
    pred = model.predict(val_data[features])
    rmse = np.sqrt(mean_squared_error(val_data['target'], pred))
    return rmse
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f"Best parameters: {study.best_params}")
print(f"Best RMSE: {study.best_value:.4f}")
best_params = study.best_params
best_params['tree_method'] = 'gpu_hist'
best_params['predictor'] = 'gpu_predictor'
best_params['objective'] = 'reg:squarederror'  # or 'binary:logistic' for classification
model = xgb.XGBRegressor(**best_params)
model.fit(train_data[features], train_data['target'])
from sklearn.model_selection import KFold
# K-Fold 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(train_data):
    X_train, X_val = train_data.iloc[train_idx][features], train_data.iloc[val_idx][features]
    y_train, y_val = train_data.iloc[train_idx]['target'], train_data.iloc[val_idx]['target']
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    cv_scores.append(np.sqrt(mean_squared_error(y_val, val_pred)))
print(f"Cross-Validation RMSE: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
# 예측
train_pred = model.predict(train_data[features])
val_pred = model.predict(val_data[features])
test_pred = model.predict(test_data[features])
train_rmse = np.sqrt(mean_squared_error(train_data['target'], train_pred))
val_rmse = np.sqrt(mean_squared_error(val_data['target'], val_pred))
test_rmse = np.sqrt(mean_squared_error(test_data['target'], test_pred))
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
# 시각화 개선
# 1. Bollinger Bands
print(f"Available columns for visualization: {train_df.columns.tolist()}")  # 디버깅용
if 'Close' in train_df.columns and 'bb_high' in train_df.columns and 'bb_low' in train_df.columns:
    plt.figure(figsize=(14, 7))
    plt.plot(train_df['datetime'], train_df['Close'], label='Close Price', color='green')
    plt.plot(train_df['datetime'], train_df['bb_high'], label='Upper Band', color='blue', linestyle='--')
    plt.plot(train_df['datetime'], train_df['bb_low'], label='Lower Band', color='red', linestyle='--')
    plt.title('Bollinger Bands (2025-07-30 20:58 KST)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Warning: 'Close', 'bb_high', or 'bb_low' columns are missing. Visualization skipped.")
# 2. Actual vs Predicted Returns
plt.figure(figsize=(14, 7))
plt.plot(val_data['datetime'], val_data['target'], label='Actual Returns', color='blue')
plt.plot(val_data['datetime'], val_pred, label='Predicted Returns', color='orange', alpha=0.7)
plt.title('Actual vs Predicted Returns (Validation Set)')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.ylim(-0.1, 0.1)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# 3. Feature Importance
feature_importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (LightGBM)')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()