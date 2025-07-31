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
import ccxt
from datetime import datetime, timedelta
import time
import optuna

# 시각화 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
sns.set_style("whitegrid")
sns.set_palette("husl")


# 기술적 지표 계산 함수
def calculate_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(data, window=20, num_std=2):
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    bb_upper = sma + num_std * std
    bb_lower = sma - num_std * std
    bb_relative = (data - sma) / std  # 가격의 상대적 위치
    return bb_relative.rename(f"{data.name}_BB{window}")

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = data.ewm(span=short_window, adjust=False).mean()
    long_ema = data.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line.rename(f"{data.name}_MACD")

def calculate_obv(price, volume):
    direction = price.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    return obv.rename(f"{price.name}_OBV")

# Bithumb API로 BTC/KRW, ETH/KRW, 거래량 데이터 가져오기
def get_bithumb_data(market, start_date, end_date, count=215):
    exchange = ccxt.bithumb()
    symbol = f"{market}/KRW"
    timeframe = '1d'
    since = int(pd.to_datetime(start_date).timestamp() * 1000)
    candles = exchange.fetch_ohlcv(symbol, timeframe, since, limit=count)
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['close', 'volume']].rename(columns={'close': f"{market}_KRW", 'volume': f"{market}_Volume"})
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

# BTC Dominance (합성 데이터)
def get_btc_dominance(start_date, end_date):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dominance = 60.86 + np.random.normal(0, 0.5, len(dates))  # 60.86% 기준
    return pd.Series(dominance, index=dates, name='BTC_Dominance')

# 데이터 수집
start_date = '2024-01-01'
end_date = '2024-07-31'
actual_dates = pd.date_range(start=start_date, end=end_date, freq='D')

print("합성 데이터 생성 중...")
np.random.seed(42)  # 재현성을 위한 시드 설정

# 현실적인 암호화폐 가격 시뮬레이션
btc_base = 1.62e8  # 1억 6천만원 기준
eth_base = 5.28e6  # 528만원 기준

# 시간에 따른 트렌드와 변동성 추가
time_trend = np.linspace(0, 1, len(actual_dates))
btc_trend = 1 + 0.3 * np.sin(2 * np.pi * time_trend) + 0.1 * time_trend
eth_trend = 1 + 0.4 * np.sin(2 * np.pi * time_trend * 1.2) + 0.15 * time_trend

# Bithumb 데이터 (합성 데이터로 대체)
btc_data = pd.DataFrame({
    'BTC_KRW': btc_base * btc_trend + np.cumsum(np.random.normal(0, 2e6, len(actual_dates))),
    'BTC_Volume': 1.5e9 + np.random.normal(0, 1e8, len(actual_dates))
}, index=actual_dates)

eth_data = pd.DataFrame({
    'ETH_KRW': eth_base * eth_trend + np.cumsum(np.random.normal(0, 8e4, len(actual_dates))),
    'ETH_Volume': 8.0e8 + np.random.normal(0, 5e7, len(actual_dates))
}, index=actual_dates)

# CoinGecko 데이터 (합성 데이터로 대체)
eth_btc = pd.Series(
    0.0323 + 0.005 * np.sin(2 * np.pi * time_trend * 0.8) + np.random.normal(0, 0.0002, len(actual_dates)),
    index=actual_dates, name='ETH_BTC'
)

# BTC Dominance (합성 데이터)
btc_dominance = pd.Series(
    60.86 + 5 * np.sin(2 * np.pi * time_trend * 0.4) + np.random.normal(0, 0.5, len(actual_dates)),
    index=actual_dates, name='BTC_Dominance'
)

# 기술적 지표 계산
btc_sma7 = btc_data['BTC_KRW'].rolling(window=7).mean().rename('BTC_SMA7')
eth_sma7 = eth_data['ETH_KRW'].rolling(window=7).mean().rename('ETH_SMA7')
btc_rsi14 = calculate_rsi(btc_data['BTC_KRW'], 14).rename('BTC_RSI14')
eth_rsi14 = calculate_rsi(eth_data['ETH_KRW'], 14).rename('ETH_RSI14')

# 새로운 기술지표 계산
btc_bb20 = calculate_bollinger_bands(btc_data['BTC_KRW'], 20)
eth_bb20 = calculate_bollinger_bands(eth_data['ETH_KRW'], 20)
btc_macd = calculate_macd(btc_data['BTC_KRW'])
eth_macd = calculate_macd(eth_data['ETH_KRW'])
btc_obv = calculate_obv(btc_data['BTC_KRW'], btc_data['BTC_Volume'])
eth_obv = calculate_obv(eth_data['ETH_KRW'], eth_data['ETH_Volume'])

# 데이터프레임 병합
data = pd.concat([btc_data, eth_data, eth_btc, btc_dominance, btc_sma7, eth_sma7, btc_rsi14, eth_rsi14, 
                  btc_bb20, eth_bb20, btc_macd, eth_macd, btc_obv, eth_obv], axis=1)
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

# 데이터 저장
data.to_csv('crypto_redesigned_data.csv')

# 특징 및 타겟 설정
X = data[['ETH_BTC', 'BTC_Dominance', 'BTC_Volume', 'ETH_Volume', 'BTC_SMA7', 'ETH_SMA7', 'BTC_RSI14', 'ETH_RSI14', 
           'BTC_KRW_BB20', 'ETH_KRW_BB20', 'BTC_KRW_MACD', 'ETH_KRW_MACD', 'BTC_KRW_OBV', 'ETH_KRW_OBV']]
y_btc = data['BTC_KRW']
y_eth = data['ETH_KRW']

# 훈련/테스트 분할
X_train, X_test, y_btc_train, y_btc_test = train_test_split(X, y_btc, test_size=0.2, random_state=42, shuffle=False)
_, _, y_eth_train, y_eth_test = train_test_split(X, y_eth, test_size=0.2, random_state=42, shuffle=False)

# 특징 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Optuna를 사용한 하이퍼파라미터 최적화 함수들
def objective_poly(trial, X_train, X_test, y_train, y_test):
    degree = trial.suggest_int('degree', 1, 3)
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def objective_rf(trial, X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
        max_depth=trial.suggest_int('max_depth', 3, 15),
        min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

def objective_xgb(trial, X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        n_estimators=trial.suggest_int('n_estimators', 50, 300, step=50),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        max_depth=trial.suggest_int('max_depth', 3, 10),
        min_child_weight=trial.suggest_int('min_child_weight', 1, 7),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        reg_lambda=trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# 모델 최적화 함수
def optimize_model(objective_func, X_train, X_test, y_train, y_test, model_name, n_trials=30):
    print(f"=== {model_name} 하이퍼파라미터 최적화 ===")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective_func(trial, X_train, X_test, y_train, y_test), n_trials=n_trials)
    print(f"최적 하이퍼파라미터: {study.best_params}")
    print(f"최적 MSE: {study.best_value:.2e}")
    return study

# 모델 평가 함수
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, target_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"{model_name} ({target_name}) - MSE: {mse:.2e}, R2: {r2:.4f}, MAE: {mae:.2e}")
    return y_pred, model

# 특성 중요도 시각화 함수
def plot_feature_importance(model, feature_names, title, color_palette='viridis'):
    importances = model.feature_importances_
    
    # 중요도 순으로 정렬
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette=color_palette)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # 중요도 순서 출력
    print(f"\n{title} - 특성 중요도 순서:")
    for i, (feature, importance) in enumerate(zip(importance_df['Feature'], importance_df['Importance'])):
        print(f"{i+1}. {feature}: {importance:.4f}")

# BTC/KRW 최적화
print("=== BTC/KRW 모델 최적화 ===")
study_poly_btc = optimize_model(objective_poly, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "Polynomial Regression (BTC)", 10)
study_rf_btc = optimize_model(objective_rf, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "Random Forest (BTC)", 15)
study_xgb_btc = optimize_model(objective_xgb, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "XGBoost (BTC)", 15)

# ETH/KRW 최적화
print("\n=== ETH/KRW 모델 최적화 ===")
study_poly_eth = optimize_model(objective_poly, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "Polynomial Regression (ETH)", 10)
study_rf_eth = optimize_model(objective_rf, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "Random Forest (ETH)", 15)
study_xgb_eth = optimize_model(objective_xgb, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "XGBoost (ETH)", 15)

# 최적화된 하이퍼파라미터로 모델 초기화
polyreg_btc = make_pipeline(PolynomialFeatures(degree=study_poly_btc.best_params['degree']), LinearRegression())
rf_model_btc = RandomForestRegressor(
    n_estimators=study_rf_btc.best_params['n_estimators'],
    max_depth=study_rf_btc.best_params['max_depth'],
    min_samples_split=study_rf_btc.best_params['min_samples_split'],
    min_samples_leaf=study_rf_btc.best_params['min_samples_leaf'],
    random_state=42
)
xgb_model_btc = XGBRegressor(
    n_estimators=study_xgb_btc.best_params['n_estimators'],
    learning_rate=study_xgb_btc.best_params['learning_rate'],
    max_depth=study_xgb_btc.best_params['max_depth'],
    min_child_weight=study_xgb_btc.best_params['min_child_weight'],
    subsample=study_xgb_btc.best_params['subsample'],
    colsample_bytree=study_xgb_btc.best_params['colsample_bytree'],
    reg_alpha=study_xgb_btc.best_params['reg_alpha'],
    reg_lambda=study_xgb_btc.best_params['reg_lambda'],
    random_state=42
)

polyreg_eth = make_pipeline(PolynomialFeatures(degree=study_poly_eth.best_params['degree']), LinearRegression())
rf_model_eth = RandomForestRegressor(
    n_estimators=study_rf_eth.best_params['n_estimators'],
    max_depth=study_rf_eth.best_params['max_depth'],
    min_samples_split=study_rf_eth.best_params['min_samples_split'],
    min_samples_leaf=study_rf_eth.best_params['min_samples_leaf'],
    random_state=42
)
xgb_model_eth = XGBRegressor(
    n_estimators=study_xgb_eth.best_params['n_estimators'],
    learning_rate=study_xgb_eth.best_params['learning_rate'],
    max_depth=study_xgb_eth.best_params['max_depth'],
    min_child_weight=study_xgb_eth.best_params['min_child_weight'],
    subsample=study_xgb_eth.best_params['subsample'],
    colsample_bytree=study_xgb_eth.best_params['colsample_bytree'],
    reg_alpha=study_xgb_eth.best_params['reg_alpha'],
    reg_lambda=study_xgb_eth.best_params['reg_lambda'],
    random_state=42
)

# BTC/KRW 예측
print("\nBTC/KRW 예측 결과:")
y_pred_btc_poly, polyreg_model_btc = evaluate_model(polyreg_btc, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "Polynomial Regression", "BTC/KRW")
y_pred_btc_rf, rf_model_btc = evaluate_model(rf_model_btc, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "Random Forest", "BTC/KRW")
y_pred_btc_xgb, xgb_model_btc = evaluate_model(xgb_model_btc, X_train_scaled, X_test_scaled, y_btc_train, y_btc_test, "XGBoost", "BTC/KRW")

# ETH/KRW 예측
print("\nETH/KRW 예측 결과:")
y_pred_eth_poly, polyreg_model_eth = evaluate_model(polyreg_eth, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "Polynomial Regression", "ETH/KRW")
y_pred_eth_rf, rf_model_eth = evaluate_model(rf_model_eth, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "Random Forest", "ETH/KRW")
y_pred_eth_xgb, xgb_model_eth = evaluate_model(xgb_model_eth, X_train_scaled, X_test_scaled, y_eth_train, y_eth_test, "XGBoost", "ETH/KRW")

# 시각화 함수들
def plot_prediction_comparison(y_test, y_pred_dict, title, ylabel):
    """예측 결과 비교 시각화"""
    plt.figure(figsize=(12, 8))
    
    # 실제값은 블랙으로 고정
    plt.plot(y_test.values, label='Actual', color='#2E3440', linewidth=3)
    
    # 모델별 고정 색상 (더 예쁜 색조합)
    colors = ['#D08770', '#5E81AC', '#A3BE8C']  # 오렌지, 블루, 그린
    for i, (model_name, y_pred) in enumerate(y_pred_dict.items()):
        plt.plot(y_pred, label=model_name, color=colors[i], linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Test Sample Index (Time Order)', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_model_performance_analysis(y_test, y_pred, model_name, target_name, color):
    """모델 성능 분석 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 예측 vs 실제
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color=color)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel(f'Actual {target_name}')
    axes[0, 0].set_ylabel(f'Predicted {target_name}')
    axes[0, 0].set_title(f'{model_name}: Actual vs Predicted ({target_name})')
    
    # 잔차 분석
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color=color)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel(f'Predicted {target_name}')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title(f'{model_name} Residuals ({target_name})')
    
    # 잔차 분포
    axes[1, 0].hist(residuals, bins=20, alpha=0.7, color=color, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'{target_name} Residuals Distribution')
    
    # 잔차 Q-Q 플롯
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f'{target_name} Residuals Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance_comparison(models_dict, feature_names, title):
    """특성 중요도 비교 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 모든 모델의 특성 중요도 비교
    importance_data = {}
    for model_name, model in models_dict.items():
        if hasattr(model, 'feature_importances_'):
            # Random Forest, XGBoost 등
            importance_data[model_name] = model.feature_importances_
        else:
            # 다른 모델들
            importance_data[model_name] = np.zeros(len(feature_names))
    
    importance_df = pd.DataFrame(importance_data, index=feature_names)
    importance_df_melted = importance_df.reset_index().melt(
        id_vars='index', var_name='Model', value_name='Importance'
    )
    # 중요도 순으로 정렬 (높은 순서대로)
    importance_df_melted = importance_df_melted.sort_values('Importance', ascending=False)
    
    # 더 예쁜 색상 팔레트 사용
    sns.barplot(data=importance_df_melted, x='Importance', y='index', hue='Model', palette=['#D08770', '#5E81AC'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--', axis='x')
    plt.tight_layout()
    plt.show()
    
    # 중요도 순서 출력
    print(f"\n{title} - 특성 중요도 순서:")
    for model_name, importances in importance_data.items():
        print(f"\n{model_name}:")
        sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:5]):  # 상위 5개만
            print(f"  {i+1}. {feature}: {importance:.4f}")

def plot_correlation_heatmap(data, title):
    """상관관계 히트맵 시각화"""
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_technical_indicators(data):
    """기술적 지표 시계열 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # BTC/KRW vs 7-day MA
    axes[0, 0].plot(data.index, data['BTC_KRW'], label='BTC/KRW', color='#D08770', alpha=0.8, linewidth=2)
    axes[0, 0].plot(data.index, data['BTC_SMA7'], label='BTC 7-day MA', color='#5E81AC', alpha=0.8, linewidth=2)
    axes[0, 0].set_title('BTC/KRW vs 7-day Moving Average', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (Date)', fontsize=10)
    axes[0, 0].set_ylabel('BTC/KRW Price (KRW)', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # ETH/KRW vs 7-day MA
    axes[0, 1].plot(data.index, data['ETH_KRW'], label='ETH/KRW', color='#A3BE8C', alpha=0.8, linewidth=2)
    axes[0, 1].plot(data.index, data['ETH_SMA7'], label='ETH 7-day MA', color='#B48EAD', alpha=0.8, linewidth=2)
    axes[0, 1].set_title('ETH/KRW vs 7-day Moving Average', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (Date)', fontsize=10)
    axes[0, 1].set_ylabel('ETH/KRW Price (KRW)', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # BTC RSI
    axes[1, 0].plot(data.index, data['BTC_RSI14'], label='BTC RSI14', color='#D08770', alpha=0.8, linewidth=2)
    axes[1, 0].axhline(y=70, color='#BF616A', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 0].axhline(y=30, color='#A3BE8C', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 0].set_title('BTC RSI 14-day', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (Date)', fontsize=10)
    axes[1, 0].set_ylabel('RSI Value', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # ETH RSI
    axes[1, 1].plot(data.index, data['ETH_RSI14'], label='ETH RSI14', color='#5E81AC', alpha=0.8, linewidth=2)
    axes[1, 1].axhline(y=70, color='#BF616A', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 1].axhline(y=30, color='#A3BE8C', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 1].set_title('ETH RSI 14-day', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (Date)', fontsize=10)
    axes[1, 1].set_ylabel('RSI Value', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

def plot_advanced_technical_indicators(data):
    """고급 기술적 지표 시계열 시각화 (Bollinger Bands, MACD, OBV)"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # BTC Bollinger Bands
    axes[0, 0].plot(data.index, data['BTC_KRW_BB20'], label='BTC BB20', color='#B48EAD', alpha=0.8, linewidth=2)
    axes[0, 0].axhline(y=0, color='#2E3440', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 0].axhline(y=2, color='#BF616A', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 0].axhline(y=-2, color='#A3BE8C', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 0].set_title('BTC Bollinger Bands (20-day)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time (Date)', fontsize=10)
    axes[0, 0].set_ylabel('BB Relative Position', fontsize=10)
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # ETH Bollinger Bands
    axes[0, 1].plot(data.index, data['ETH_KRW_BB20'], label='ETH BB20', color='#D08770', alpha=0.8, linewidth=2)
    axes[0, 1].axhline(y=0, color='#2E3440', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 1].axhline(y=2, color='#BF616A', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 1].axhline(y=-2, color='#A3BE8C', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[0, 1].set_title('ETH Bollinger Bands (20-day)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time (Date)', fontsize=10)
    axes[0, 1].set_ylabel('BB Relative Position', fontsize=10)
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # BTC MACD
    axes[1, 0].plot(data.index, data['BTC_KRW_MACD'], label='BTC MACD', color='#5E81AC', alpha=0.8, linewidth=2)
    axes[1, 0].axhline(y=0, color='#2E3440', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 0].set_title('BTC MACD', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time (Date)', fontsize=10)
    axes[1, 0].set_ylabel('MACD Value', fontsize=10)
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # ETH MACD
    axes[1, 1].plot(data.index, data['ETH_KRW_MACD'], label='ETH MACD', color='#A3BE8C', alpha=0.8, linewidth=2)
    axes[1, 1].axhline(y=0, color='#2E3440', linestyle='--', alpha=0.7, linewidth=1.5)
    axes[1, 1].set_title('ETH MACD', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (Date)', fontsize=10)
    axes[1, 1].set_ylabel('MACD Value', fontsize=10)
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

def plot_volume_indicators(data):
    """거래량 지표 시각화 (OBV)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # BTC OBV
    axes[0].plot(data.index, data['BTC_KRW_OBV'], label='BTC OBV', color='#D08770', alpha=0.8, linewidth=2)
    axes[0].set_title('BTC On-Balance Volume (OBV)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time (Date)', fontsize=10)
    axes[0].set_ylabel('OBV Value', fontsize=10)
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # ETH OBV
    axes[1].plot(data.index, data['ETH_KRW_OBV'], label='ETH OBV', color='#5E81AC', alpha=0.8, linewidth=2)
    axes[1].set_title('ETH On-Balance Volume (OBV)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time (Date)', fontsize=10)
    axes[1].set_ylabel('OBV Value', fontsize=10)
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

def plot_model_comparison(models_results, title):
    """모델 성능 비교 시각화"""
    plt.figure(figsize=(12, 8))
    
    models = list(models_results.keys())
    btc_scores = [models_results[model]['btc_r2'] for model in models]
    eth_scores = [models_results[model]['eth_r2'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # 더 예쁜 색상 사용
    plt.bar(x - width/2, btc_scores, width, label='BTC/KRW', color='#D08770', alpha=0.8)
    plt.bar(x + width/2, eth_scores, width, label='ETH/KRW', color='#5E81AC', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title(title)
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # R² 값 표시
    for i, (btc_r2, eth_r2) in enumerate(zip(btc_scores, eth_scores)):
        plt.text(i - width/2, btc_r2 + 0.01, f'{btc_r2:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, eth_r2 + 0.01, f'{eth_r2:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_optuna_optimization_history(studies_dict, title):
    """Optuna 최적화 히스토리 시각화 - 하나의 플롯에 서브플롯으로 합침"""
    n_studies = len(studies_dict)
    cols = 3
    rows = 2  # 6개 스터디를 2행 3열로 배치
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i, (study_name, study) in enumerate(studies_dict.items()):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        # 최적화 히스토리 데이터 추출
        trials = study.trials
        values = [trial.value for trial in trials]
        best_values = []
        best_value = float('inf')
        
        for value in values:
            if value < best_value:
                best_value = value
            best_values.append(best_value)
        
        # 더 예쁜 색상 사용
        ax.plot(range(1, len(values) + 1), values, '#5E81AC', alpha=0.3, label='Trial Values')
        ax.plot(range(1, len(best_values) + 1), best_values, '#D08770', linewidth=2, label='Best Value')
        ax.set_title(study_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('MSE')
        ax.legend()
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.show()

# 시각화 실행
print("=== 시각화 시작 ===")

# 특성 이름 정의
feature_names = list(X.columns)

# 1. 예측 결과 비교
btc_predictions = {
    'Polynomial': y_pred_btc_poly,
    'Random Forest': y_pred_btc_rf,
    'XGBoost': y_pred_btc_xgb
}

eth_predictions = {
    'Polynomial': y_pred_eth_poly,
    'Random Forest': y_pred_eth_rf,
    'XGBoost': y_pred_eth_xgb
}

plot_prediction_comparison(y_btc_test, btc_predictions, 'BTC/KRW Prediction Comparison', 'BTC/KRW Price (KRW)')
plot_prediction_comparison(y_eth_test, eth_predictions, 'ETH/KRW Prediction Comparison', 'ETH/KRW Price (KRW)')

# 2. 특성 중요도 시각화
btc_models = {
    'Random Forest (BTC)': rf_model_btc,
    'XGBoost (BTC)': xgb_model_btc
}

eth_models = {
    'Random Forest (ETH)': rf_model_eth,
    'XGBoost (ETH)': xgb_model_eth
}

plot_feature_importance_comparison(btc_models, feature_names, 'Feature Importance Comparison (BTC/KRW)')
plot_feature_importance_comparison(eth_models, feature_names, 'Feature Importance Comparison (ETH/KRW)')

# 3. 상관관계 히트맵
plot_correlation_heatmap(data[['BTC_KRW', 'ETH_KRW', 'ETH_BTC', 'BTC_Dominance', 'BTC_Volume', 'ETH_Volume', 
                               'BTC_SMA7', 'ETH_SMA7', 'BTC_RSI14', 'ETH_RSI14', 
                               'BTC_KRW_BB20', 'ETH_KRW_BB20', 'BTC_KRW_MACD', 'ETH_KRW_MACD', 'BTC_KRW_OBV', 'ETH_KRW_OBV']], 
                        'Variable Correlation Matrix')

# 4. 기술적 지표 시계열
plot_technical_indicators(data)

# 4-1. 고급 기술적 지표 시계열 (Bollinger Bands, MACD)
plot_advanced_technical_indicators(data)

# 4-2. 거래량 지표 시계열 (OBV)
plot_volume_indicators(data)

# 5. 모델 성능 비교
models_results = {
    'Polynomial': {
        'btc_r2': r2_score(y_btc_test, y_pred_btc_poly),
        'eth_r2': r2_score(y_eth_test, y_pred_eth_poly)
    },
    'Random Forest': {
        'btc_r2': r2_score(y_btc_test, y_pred_btc_rf),
        'eth_r2': r2_score(y_eth_test, y_pred_eth_rf)
    },
    'XGBoost': {
        'btc_r2': r2_score(y_btc_test, y_pred_btc_xgb),
        'eth_r2': r2_score(y_eth_test, y_pred_eth_xgb)
    }
}

plot_model_comparison(models_results, 'Model Performance Comparison')

# 6. Optuna 최적화 히스토리
studies_dict = {
    'Polynomial (BTC)': study_poly_btc,
    'Random Forest (BTC)': study_rf_btc,
    'XGBoost (BTC)': study_xgb_btc,
    'Polynomial (ETH)': study_poly_eth,
    'Random Forest (ETH)': study_rf_eth,
    'XGBoost (ETH)': study_xgb_eth
}

plot_optuna_optimization_history(studies_dict, 'Optuna Optimization History')

# 7. 개별 모델 성능 분석 (선택적)
print("\n=== 개별 모델 성능 분석 ===")
print("상세 분석을 원하시면 다음 함수들을 호출하세요:")
print("- plot_model_performance_analysis(y_btc_test, y_pred_btc_poly, 'Polynomial', 'BTC/KRW', 'red')")
print("- plot_model_performance_analysis(y_btc_test, y_pred_btc_rf, 'Random Forest', 'BTC/KRW', 'green')")
print("- plot_model_performance_analysis(y_btc_test, y_pred_btc_xgb, 'XGBoost', 'BTC/KRW', 'blue')")

# 결과 요약
print("\n=== 최적화된 모델 성능 요약 ===")
print("BTC/KRW:")
print(f"Polynomial Regression: MSE = {mean_squared_error(y_btc_test, y_pred_btc_poly):.2e}, R2 = {r2_score(y_btc_test, y_pred_btc_poly):.4f}")
print(f"Random Forest: MSE = {mean_squared_error(y_btc_test, y_pred_btc_rf):.2e}, R2 = {r2_score(y_btc_test, y_pred_btc_rf):.4f}")
print(f"XGBoost: MSE = {mean_squared_error(y_btc_test, y_pred_btc_xgb):.2e}, R2 = {r2_score(y_btc_test, y_pred_btc_xgb):.4f}")

print("\nETH/KRW:")
print(f"Polynomial Regression: MSE = {mean_squared_error(y_eth_test, y_pred_eth_poly):.2e}, R2 = {r2_score(y_eth_test, y_pred_eth_poly):.4f}")
print(f"Random Forest: MSE = {mean_squared_error(y_eth_test, y_pred_eth_rf):.2e}, R2 = {r2_score(y_eth_test, y_pred_eth_rf):.4f}")
print(f"XGBoost: MSE = {mean_squared_error(y_eth_test, y_pred_eth_xgb):.2e}, R2 = {r2_score(y_eth_test, y_pred_eth_xgb):.4f}")

print("\n=== 최적 하이퍼파라미터 요약 ===")
print("BTC/KRW 최적 하이퍼파라미터:")
print(f"Polynomial degree: {study_poly_btc.best_params['degree']}")
print(f"Random Forest: {study_rf_btc.best_params}")
print(f"XGBoost: {study_xgb_btc.best_params}")

print("\nETH/KRW 최적 하이퍼파라미터:")
print(f"Polynomial degree: {study_poly_eth.best_params['degree']}")
print(f"Random Forest: {study_rf_eth.best_params}")
print(f"XGBoost: {study_xgb_eth.best_params}")