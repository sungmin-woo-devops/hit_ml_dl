"""
공통 유틸리티 함수들
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def setup_environment():
    """환경 설정"""
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    pd.options.display.float_format = '{:.2f}'.format
    
    # 데이터 폴더 생성
    data_dir = get_data_paths()['data_dir']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"데이터 폴더 생성: {data_dir}")

def get_data_paths():
    """데이터 경로 설정"""
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    
    return {
        'base_dir': base_dir,
        'data_dir': data_dir,
        'before_feature_path': os.path.join(data_dir, "merged_data.csv"),
        'after_feature_path': os.path.join(data_dir, "merged_data_features.csv"),
        'cleaned_data_path': os.path.join(data_dir, "merged_data_features_cleaned.csv"),
        'pkl_data_path': os.path.join(data_dir, "merged_data_features.pkl")
    }

def validate_dataframe(df, required_columns=None):
    """데이터프레임 유효성 검사"""
    if df is None or df.empty:
        return False, "데이터프레임이 비어있습니다."
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"필수 컬럼이 누락되었습니다: {missing_columns}"
    
    return True, "유효한 데이터프레임입니다."

def get_price_column(df):
    """가격 컬럼 찾기"""
    price_cols = [col for col in df.columns if any(x in col.lower() for x in ['btc', 'price', 'usd_eur'])]
    return price_cols[0] if price_cols else None

def get_technical_indicators(df):
    """기술지표 컬럼들 찾기"""
    indicators = {}
    
    # RSI
    rsi_cols = [col for col in df.columns if 'rsi' in col.lower()]
    indicators['rsi'] = rsi_cols[0] if rsi_cols else None
    
    # 볼린저 밴드
    bb_cols = [col for col in df.columns if 'bb_percent' in col.lower()]
    indicators['bb_percent'] = bb_cols[0] if bb_cols else None
    
    # 이동평균 비율
    ma_cols = [col for col in df.columns if 'ma_ratio' in col.lower()]
    indicators['ma_ratio'] = ma_cols[0] if ma_cols else None
    
    # 로그 수익률
    log_cols = [col for col in df.columns if 'log_returns' in col.lower()]
    indicators['log_returns'] = log_cols[0] if log_cols else None
    
    # 거래량 비율
    vol_cols = [col for col in df.columns if 'vol_ratio' in col.lower()]
    indicators['vol_ratio'] = vol_cols[0] if vol_cols else None
    
    return indicators 