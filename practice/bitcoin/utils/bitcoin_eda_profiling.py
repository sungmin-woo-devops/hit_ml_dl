import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./data/merged_data_features.csv')

print(f"원본 데이터 shape: {df.shape}")
print(f"컬럼 수: {len(df.columns)}")

if 'date' not in df.columns:
    df['date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
else:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

if 'btc_usd' not in df.columns and 'BTC_EUR' in df.columns and 'USD_EUR' in df.columns:
    df['btc_usd'] = df['BTC_EUR'] / df['USD_EUR']

df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.day_name()
df['quarter'] = df['date'].dt.quarter

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

df['season'] = df['month'].apply(get_season)

if 'btc_usd' in df.columns:
    df['btc_price_change'] = df['btc_usd'].diff()
    df['btc_price_change_pct'] = df['btc_usd'].pct_change() * 100
    
    df['btc_ma_5'] = df['btc_usd'].rolling(5).mean()
    df['btc_ma_10'] = df['btc_usd'].rolling(10).mean()
    df['btc_ma_30'] = df['btc_usd'].rolling(30).mean()
    
    df['btc_volatility_5'] = df['btc_price_change_pct'].rolling(5).std()
    df['btc_volatility_20'] = df['btc_price_change_pct'].rolling(20).std()

volume_cols = [col for col in df.columns if 'volume' in col.lower()]
if volume_cols:
    main_volume = volume_cols[0]
    df['volume_ma_5'] = df[main_volume].rolling(5).mean()
    df['volume_ma_20'] = df[main_volume].rolling(20).mean()
    
    volume_quantiles = df[main_volume].quantile([0.2, 0.4, 0.6, 0.8])
    df['volume_level'] = pd.cut(df[main_volume], 
                               bins=[0, volume_quantiles[0.2], volume_quantiles[0.4], 
                                    volume_quantiles[0.6], volume_quantiles[0.8], float('inf')],
                               labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])

if 'btc_usd' in df.columns:
    price_quantiles = df['btc_usd'].quantile([0.2, 0.4, 0.6, 0.8])
    df['price_level'] = pd.cut(df['btc_usd'], 
                              bins=[0, price_quantiles[0.2], price_quantiles[0.4], 
                                   price_quantiles[0.6], price_quantiles[0.8], float('inf')],
                              labels=['Very_Low', 'Low', 'Medium', 'High', 'Very_High'])

if 'log_returns' in df.columns:
    df['market_direction'] = np.where(df['log_returns'] > 0.02, 'Strong_Up',
                             np.where(df['log_returns'] > 0, 'Up',
                             np.where(df['log_returns'] < -0.02, 'Strong_Down', 
                             np.where(df['log_returns'] < 0, 'Down', 'Flat'))))

df['is_monday'] = (df['weekday'] == 'Monday').astype(int)
df['is_friday'] = (df['weekday'] == 'Friday').astype(int)
df['is_month_start'] = (df['day'] <= 5).astype(int)
df['is_month_end'] = (df['day'] >= 25).astype(int)

fx_pairs = []
if 'USD_EUR' in df.columns:
    fx_pairs.append('USD_EUR')
if 'USD_JPY' in df.columns:
    fx_pairs.append('USD_JPY')
if 'USD_CNY' in df.columns:
    fx_pairs.append('USD_CNY')
if 'USD_KRW' in df.columns:
    fx_pairs.append('USD_KRW')

for pair in fx_pairs:
    df[f'{pair}_change'] = df[pair].pct_change() * 100
    df[f'{pair}_volatility'] = df[f'{pair}_change'].rolling(20).std()

if 'XAU_USD' in df.columns:
    df['gold_change_pct'] = df['XAU_USD'].pct_change() * 100
    df['gold_volatility'] = df['gold_change_pct'].rolling(20).std()

btc_cols = [col for col in df.columns if 'btc' in col.lower()]
if len(btc_cols) > 1:
    df['btc_eur_jpy_ratio'] = 1
    if 'BTC_EUR' in df.columns and 'BTC_JPY' in df.columns:
        df['btc_eur_jpy_ratio'] = df['BTC_EUR'] / (df['BTC_JPY'] / 1000)

print(f"전처리 후 shape: {df.shape}")

missing_info = df.isnull().sum()
missing_cols = missing_info[missing_info > 0]
if len(missing_cols) > 0:
    print("결측값 존재:")
    print(missing_cols)
else:
    print("결측값 없음")

print(f"분석 기간: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
print(f"총 데이터 수: {len(df)}일")

profile = ProfileReport(
    df,
    title="Bitcoin and Forex Data Analysis Report",
    dataset={
        "description": f"Bitcoin, Forex, and Gold data comprehensive analysis ({df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')})",
        "copyright_holder": "Crypto Analysis Team",
        "copyright_year": "2025"
    },
    variables={
        "descriptions": {
            "date": "거래 날짜",
            "btc_usd": "BTC/USD 가격",
            "USD_EUR": "USD/EUR 환율",
            "USD_JPY": "USD/JPY 환율", 
            "USD_CNY": "USD/CNY 환율",
            "USD_KRW": "USD/KRW 환율",
            "XAU_USD": "금/USD 가격",
            "BTC_EUR": "BTC/EUR 가격",
            "BTC_JPY": "BTC/JPY 가격",
            "BTC_Volume": "BTC 거래량",
            "log_returns": "로그 수익률",
            "rsi": "RSI 지표",
            "bb_percent": "볼린저 밴드 위치",
            "ma_ratio": "이동평균 비율",
            "vol_ratio": "거래량 비율",
            "price_vol_corr": "가격-거래량 상관관계",
            "year": "연도",
            "month": "월",
            "quarter": "분기",
            "weekday": "요일",
            "season": "계절",
            "market_direction": "시장 방향",
            "price_level": "가격 수준",
            "volume_level": "거래량 수준"
        }
    },
    minimal=False,
    explorative=True
)

print("HTML 레포트 생성 중...")
profile.to_file("bitcoin_forex_profiling_report.html")

print("JSON 레포트 생성 중...")
profile.to_file("bitcoin_forex_profiling_report.json")

summary_stats = df.describe(include='all')
summary_stats.to_csv('bitcoin_forex_summary_statistics.csv', encoding='utf-8-sig')

numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
correlation_matrix.to_csv('bitcoin_forex_correlation_matrix.csv', encoding='utf-8-sig')

df.to_csv('bitcoin_forex_processed_data.csv', index=False, encoding='utf-8-sig')

weekday_stats = df.groupby('weekday').agg({
    'log_returns': ['mean', 'std', 'min', 'max'],
    'btc_usd': 'mean'
}).round(4)

monthly_stats = df.groupby(['year', 'month']).agg({
    'log_returns': ['mean', 'std', 'count'],
    'btc_usd': ['first', 'last', 'min', 'max']
}).round(4)

weekday_stats.to_csv('bitcoin_weekday_stats.csv', encoding='utf-8-sig')
monthly_stats.to_csv('bitcoin_monthly_stats.csv', encoding='utf-8-sig')

print("분석 완료")
print(f"분석 기간: {df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}")
print(f"총 거래일 수: {len(df)}일")
print(f"총 변수 수: {len(df.columns)}개")
print(f"수치형 변수: {len(df.select_dtypes(include=[np.number]).columns)}개")
print(f"범주형 변수: {len(df.select_dtypes(include=['object', 'category']).columns)}개")

if 'btc_usd' in df.columns:
    print(f"BTC/USD 최고가: ${df['btc_usd'].max():,.2f}")
    print(f"BTC/USD 최저가: ${df['btc_usd'].min():,.2f}")
    print(f"BTC/USD 평균가: ${df['btc_usd'].mean():,.2f}")
    price_range = ((df['btc_usd'].max() - df['btc_usd'].min()) / df['btc_usd'].min() * 100)
    print(f"가격 변동폭: {price_range:.1f}%")

if 'log_returns' in df.columns:
    returns = df['log_returns'].dropna()
    print(f"평균 일 수익률: {returns.mean()*100:.2f}%")
    print(f"일 수익률 표준편차: {returns.std()*100:.2f}%")
    print(f"최고 일 수익률: {returns.max()*100:.2f}%")
    print(f"최저 일 수익률: {returns.min()*100:.2f}%")
    
    up_days = len(df[df['log_returns'] > 0])
    down_days = len(df[df['log_returns'] < 0])
    flat_days = len(df[df['log_returns'] == 0])
    
    print(f"상승일: {up_days}일 ({up_days/len(df)*100:.1f}%)")
    print(f"하락일: {down_days}일 ({down_days/len(df)*100:.1f}%)")
    print(f"보합일: {flat_days}일 ({flat_days/len(df)*100:.1f}%)")

if 'season' in df.columns and 'log_returns' in df.columns:
    seasonal_returns = df.groupby('season')['log_returns'].mean().sort_values(ascending=False)
    print("계절별 성과:")
    for season, ret in seasonal_returns.items():
        if not pd.isna(ret):
            print(f"{season}: {ret*100:.2f}%")

if 'weekday' in df.columns and 'log_returns' in df.columns:
    weekday_returns = df.groupby('weekday')['log_returns'].mean()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print("요일별 성과:")
    for day in weekday_order:
        if day in weekday_returns.index:
            print(f"{day}: {weekday_returns[day]*100:.2f}%")

print("생성된 파일:")
print("bitcoin_forex_profiling_report.html - 종합 프로파일링 레포트")
print("bitcoin_forex_profiling_report.json - JSON 레포트")
print("bitcoin_forex_summary_statistics.csv - 요약 통계")
print("bitcoin_forex_correlation_matrix.csv - 상관관계 행렬")
print("bitcoin_forex_processed_data.csv - 전처리된 데이터")
print("bitcoin_weekday_stats.csv - 요일별 통계")
print("bitcoin_monthly_stats.csv - 월별 통계")

print("EDA 완료")
