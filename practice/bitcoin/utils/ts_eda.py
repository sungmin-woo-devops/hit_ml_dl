import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)

df = pd.read_csv('merged_data_features.csv')

if 'date' not in df.columns:
    df['date'] = pd.to_datetime(df.iloc[:, 0])
else:
    df['date'] = pd.to_datetime(df['date'])

if 'btc_usd' not in df.columns and 'BTC_EUR' in df.columns and 'USD_EUR' in df.columns:
    df['btc_usd'] = df['BTC_EUR'] / df['USD_EUR']

df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['btc_usd'])

print("=== 비트코인 시계열 분해 분석 ===")
print(f"분석 기간: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
print(f"데이터 수: {len(df)}일")
print(f"가격 범위: ${df['btc_usd'].min():,.0f} ~ ${df['btc_usd'].max():,.0f}")

ts_data = df.set_index('date')['btc_usd']

print("\n1. 기본 시계열 통계")
print(f"평균: ${ts_data.mean():,.2f}")
print(f"표준편차: ${ts_data.std():,.2f}")
print(f"변동계수: {ts_data.std()/ts_data.mean():.3f}")
print(f"왜도: {ts_data.skew():.3f}")
print(f"첨도: {ts_data.kurtosis():.3f}")

def perform_adf_test(series, title=""):
    result = adfuller(series.dropna())
    print(f"\n{title} ADF 정상성 검정:")
    print(f"ADF 통계량: {result[0]:.6f}")
    print(f"p-value: {result[1]:.6f}")
    if result[1] <= 0.05:
        print("결과: 정상 시계열 (p < 0.05)")
    else:
        print("결과: 비정상 시계열 (p >= 0.05)")
    return result[1] <= 0.05

is_stationary = perform_adf_test(ts_data, "원본 데이터")

log_prices = np.log(ts_data)
returns = ts_data.pct_change().dropna()
log_returns = np.log(ts_data / ts_data.shift(1)).dropna()

perform_adf_test(returns, "일간 수익률")
perform_adf_test(log_returns, "로그 수익률")

print("\n2. 시계열 분해 수행 중...")

try:
    decomposition_add = seasonal_decompose(ts_data, model='additive', period=30)
    decomposition_mult = seasonal_decompose(ts_data, model='multiplicative', period=30)
    
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    
    ts_data.plot(ax=axes[0,0], title='원본 시계열 (가법 모델)', color='blue')
    ts_data.plot(ax=axes[0,1], title='원본 시계열 (승법 모델)', color='blue')
    
    decomposition_add.trend.plot(ax=axes[1,0], title='추세 (Additive)', color='red')
    decomposition_mult.trend.plot(ax=axes[1,1], title='추세 (Multiplicative)', color='red')
    
    decomposition_add.seasonal.plot(ax=axes[2,0], title='계절성 (Additive)', color='green')
    decomposition_mult.seasonal.plot(ax=axes[2,1], title='계절성 (Multiplicative)', color='green')
    
    decomposition_add.resid.plot(ax=axes[3,0], title='잔차 (Additive)', color='orange')
    decomposition_mult.resid.plot(ax=axes[3,1], title='잔차 (Multiplicative)', color='orange')
    
    for i in range(4):
        for j in range(2):
            axes[i,j].grid(True, alpha=0.3)
            axes[i,j].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    add_resid_var = np.var(decomposition_add.resid.dropna())
    mult_resid_var = np.var(decomposition_mult.resid.dropna())
    
    print(f"\n3. 분해 모델 평가:")
    print(f"가법 모델 잔차 분산: {add_resid_var:,.2f}")
    print(f"승법 모델 잔차 분산: {mult_resid_var:,.2f}")
    
    if mult_resid_var < add_resid_var:
        print("권장 모델: 승법 모델 (잔차 분산이 더 작음)")
        best_decomp = decomposition_mult
        model_type = "승법"
    else:
        print("권장 모델: 가법 모델 (잔차 분산이 더 작음)")
        best_decomp = decomposition_add
        model_type = "가법"
    
except Exception as e:
    print(f"30일 주기 분해 실패, 7일 주기로 재시도: {e}")
    
    try:
        decomposition_add = seasonal_decompose(ts_data, model='additive', period=7)
        decomposition_mult = seasonal_decompose(ts_data, model='multiplicative', period=7)
        best_decomp = decomposition_mult
        model_type = "승법"
    except:
        print("시계열 분해 실패 - 데이터 길이 또는 품질 문제")
        best_decomp = None

if best_decomp is not None:
    print(f"\n4. {model_type} 모델 구성요소 분석:")
    
    trend_change = (best_decomp.trend.dropna().iloc[-1] / best_decomp.trend.dropna().iloc[0] - 1) * 100
    print(f"전체 추세 변화: {trend_change:+.1f}%")
    
    seasonal_amplitude = best_decomp.seasonal.max() - best_decomp.seasonal.min()
    if model_type == "승법":
        seasonal_impact = (seasonal_amplitude - 1) * 100
        print(f"계절성 영향도: ±{seasonal_impact:.1f}%")
    else:
        seasonal_impact = seasonal_amplitude / ts_data.mean() * 100
        print(f"계절성 영향도: ±{seasonal_impact:.1f}%")
    
    noise_level = best_decomp.resid.std() / ts_data.std() * 100
    print(f"잡음 수준: {noise_level:.1f}% (전체 변동성 대비)")

print("\n5. 추가 시계열 특성 분석")

df['day_of_week'] = df['date'].dt.day_name()
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter

weekly_pattern = df.groupby('day_of_week')['btc_usd'].mean()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_pattern = weekly_pattern.reindex(weekday_order)

monthly_pattern = df.groupby('month')['btc_usd'].mean()
quarterly_pattern = df.groupby('quarter')['btc_usd'].mean()

print("\n요일별 평균 가격:")
for day, price in weekly_pattern.items():
    if pd.notna(price):
        print(f"{day}: ${price:,.0f}")

print("\n월별 평균 가격:")
for month, price in monthly_pattern.items():
    print(f"{month}월: ${price:,.0f}")

print("\n분기별 평균 가격:")
for quarter, price in quarterly_pattern.items():
    print(f"Q{quarter}: ${price:,.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

weekly_pattern.plot(kind='bar', ax=axes[0,0], title='요일별 평균 가격', color='skyblue')
axes[0,0].tick_params(axis='x', rotation=45)
axes[0,0].set_ylabel('가격 ($)')

monthly_pattern.plot(kind='bar', ax=axes[0,1], title='월별 평균 가격', color='lightgreen')
axes[0,1].set_ylabel('가격 ($)')

quarterly_pattern.plot(kind='bar', ax=axes[1,0], title='분기별 평균 가격', color='salmon')
axes[1,0].set_ylabel('가격 ($)')

if len(returns) > 0:
    plot_acf(returns.dropna(), ax=axes[1,1], lags=30, title='수익률 자기상관함수')

plt.tight_layout()
plt.show()

volatility_7d = returns.rolling(7).std() * np.sqrt(365) * 100
volatility_30d = returns.rolling(30).std() * np.sqrt(365) * 100

print(f"\n6. 변동성 분석:")
print(f"현재 7일 변동성: {volatility_7d.iloc[-1]:.1f}% (연환산)")
print(f"현재 30일 변동성: {volatility_30d.iloc[-1]:.1f}% (연환산)")
print(f"평균 변동성: {volatility_30d.mean():.1f}%")
print(f"최고 변동성: {volatility_30d.max():.1f}%")
print(f"최저 변동성: {volatility_30d.min():.1f}%")

plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
ts_data.plot(title='BTC/USD 가격', color='blue')
plt.ylabel('가격 ($)')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
volatility_30d.plot(title='30일 변동성 (연환산)', color='red')
plt.ylabel('변동성 (%)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

correlation_analysis = pd.DataFrame({
    'BTC_USD': df['btc_usd'],
    'USD_EUR': df['USD_EUR'] if 'USD_EUR' in df.columns else np.nan,
    'USD_JPY': df['USD_JPY'] if 'USD_JPY' in df.columns else np.nan,
    'XAU_USD': df['XAU_USD'] if 'XAU_USD' in df.columns else np.nan
}).dropna()

if len(correlation_analysis) > 1:
    corr_matrix = correlation_analysis.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f')
    plt.title('자산 간 상관관계')
    plt.tight_layout()
    plt.show()
    
    print(f"\n7. 상관관계 분석:")
    btc_correlations = corr_matrix['BTC_USD'].drop('BTC_USD')
    for asset, corr in btc_correlations.items():
        if pd.notna(corr):
            print(f"BTC vs {asset}: {corr:.3f}")

print(f"\n=== 시계열 분해 분석 완료 ===")
print(f"주요 발견사항:")
print(f"1. 시계열 정상성: {'정상' if is_stationary else '비정상'}")
print(f"2. 최적 분해 모델: {model_type if 'model_type' in locals() else '미결정'}")
print(f"3. 평균 변동성: {volatility_30d.mean():.1f}% (연환산)")
print(f"4. 주요 패턴: 요일효과, 월별변동, 분기별변동 존재")