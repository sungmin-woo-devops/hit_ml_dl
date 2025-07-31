"""
기술지표 생성 모듈
"""

import pandas as pd
import numpy as np
from .utils import get_data_paths, validate_dataframe

class FeatureEngineer:
    """기술지표 생성 클래스"""
    
    def __init__(self):
        self.paths = get_data_paths()
    
    def create_core_features(self, df):
        """핵심 기술지표 생성"""
        # 데이터프레임 유효성 검사
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            print(f"데이터프레임 오류: {message}")
            return df
        
        data = df.copy()

        # log return
        if 'log_returns' not in data.columns:
            price_col = self._get_price_column(data)
            if price_col:
                data['log_returns'] = np.log(data[price_col] / data[price_col].shift(1))

        # RSI (14)
        if 'rsi' not in data.columns:
            price_col = self._get_price_column(data)
            if price_col:
                delta = data[price_col].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))

        # 볼린저 밴드 %B
        if 'bb_percent' not in data.columns:
            price_col = self._get_price_column(data)
            if price_col:
                ma20 = data[price_col].rolling(20).mean()
                std20 = data[price_col].rolling(20).std()
                data['bb_percent'] = (data[price_col] - (ma20 - 2 * std20)) / (4 * std20)

        # 이동평균 비율
        if 'ma_ratio' not in data.columns:
            price_col = self._get_price_column(data)
            if price_col:
                ma20 = data[price_col].rolling(20).mean()
                ma50 = data[price_col].rolling(50).mean()
                data['ma_ratio'] = ma20 / ma50

        # 거래량 비율
        if 'vol_ratio' not in data.columns:
            volume_col = self._get_volume_column(data)
            if volume_col:
                vol_ma = data[volume_col].rolling(20).mean()
                data['vol_ratio'] = data[volume_col] / vol_ma

        # 결측치 제거
        print("기술지표 생성 후 결측치:")
        print(data.isnull().sum().to_string())
        data = data.dropna().reset_index(drop=True)

        # 저장
        data.to_csv(self.paths['after_feature_path'], index=False, encoding='utf-8')
        return data
    
    def _get_price_column(self, df):
        """가격 컬럼 찾기"""
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['btc', 'price', 'usd_eur'])]
        return price_cols[0] if price_cols else None
    
    def _get_volume_column(self, df):
        """거래량 컬럼 찾기"""
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        return volume_cols[0] if volume_cols else None
    
    def handle_missing_values(self, df):
        """결측값 처리"""
        data = df.copy()
        original_length = len(data)

        # 1. 로그 수익률 (1개 결측) - 제거 (계산 불가능)
        if 'log_returns' in data.columns:
            data = data.dropna(subset=['log_returns'])
            
        # 2. RSI (13개 결측) - 부분 채우기 + 제거
        if 'rsi' in data.columns:
            # RSI가 계산되기 시작하는 시점부터 유효한 데이터로 간주
            # 처음 10개는 제거하고, 나머지는 전진 채우기
            rsi_start_idx = data['rsi'].first_valid_index()
            if rsi_start_idx is not None and rsi_start_idx > 10:
                data = data.iloc[10:].reset_index(drop=True)
                data['rsi'] = data['rsi'].fillna(method='ffill')

        # 3. 볼린저 밴드 (19개 결측) - 이동평균으로 추정
        if 'bb_percent' in data.columns:
            bb_missing_mask = data['bb_percent'].isnull()
            data.loc[bb_missing_mask, 'bb_percent'] = 0.5

        # 4. 이동평균 비율 (49개 결측) - 점진적 처리
        if 'ma_ratio' in data.columns:
            ma_missing_mask = data['ma_ratio'].isnull()
            data.loc[ma_missing_mask, 'ma_ratio'] = 1.0
        
        # 5. 거래량 비율 (19개) - 1.0으로 초기화 (중립)
        if 'vol_ratio' in data.columns:
            vol_missing_mask = data['vol_ratio'].isnull()
            data.loc[vol_missing_mask, 'vol_ratio'] = 1.0
        
        # 6. 가격-거래량 상관관계 - 0으로 초기화 (중립)
        if 'price_vol_corr' in data.columns:
            corr_missing_mask = data['price_vol_corr'].isnull()
            data.loc[corr_missing_mask, 'price_vol_corr'] = 0.0

        print(f"결측값 처리 완료: {original_length} -> {len(data)} 행")
        return data
    
    def update_original_csv(self, original_df, features_df):
        """원본 CSV에 기술지표 컬럼 추가"""
        feature_columns = ['log_returns', 'rsi', 'bb_percent', 'ma_ratio', 'vol_ratio', 'price_vol_corr']
        original_df['date'] = pd.to_datetime(original_df.iloc[:, 0])
        
        for col in feature_columns:
            if col in features_df.columns:
                merge_data = features_df[['date', col]].copy()
                original_df = original_df.merge(merge_data, on='date', how='left')
        
        original_df.to_csv(self.paths['after_feature_path'], index=False)
        print(f"기술지표 추가 완료: {original_df.shape}")
        print(f"추가된 컬럼: {feature_columns}")
        
        return original_df
    
    def create_all_features(self, df):
        """모든 기술지표 생성"""
        print("기술지표 생성 시작...")
        
        # 핵심 기술지표 생성
        df_with_features = self.create_core_features(df)
        
        # 결측값 처리
        df_processed = self.handle_missing_values(df_with_features)
        
        print("기술지표 생성 완료!")
        return df_processed 