"""
데이터 처리 모듈
"""

import pandas as pd
import numpy as np
from .utils import get_data_paths, validate_dataframe, get_technical_indicators
import os

class DataProcessor:
    """데이터 처리 클래스"""
    
    def __init__(self):
        self.paths = get_data_paths()
    
    def load_and_clean_data(self):
        """데이터 로드 및 정리"""
        try:
            # 여러 데이터 파일 시도
            data_files = [
                self.paths['cleaned_data_path'],
                self.paths['after_feature_path'],
                os.path.join(self.paths['data_dir'], "crypto_forex_data.csv")
            ]
            
            for file_path in data_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    print(f"데이터 로드 성공: {file_path}")
                    return df
            
            print("사용 가능한 데이터 파일을 찾을 수 없습니다.")
            return None
            
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return None
    
    def analyze_correlations(self, df):
        """상관관계 분석"""
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Unnamed: 0', 'index']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) < 2:
            print("상관관계 분석을 위한 충분한 수치형 컬럼이 없습니다.")
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        # 강한 상관관계 찾기
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= 0.7:
                    strong_correlations.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        print(f"강한 상관관계 (|상관계수| >= 0.7): {len(strong_correlations)}개")
        for pair in strong_correlations[:10]:
            direction = "양의" if pair['correlation'] > 0 else "음의"
            print(f"  {direction} {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")
        
        return corr_matrix
    
    def get_data_summary(self, df):
        """데이터 요약 정보"""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
        }
        
        # 기술지표 정보
        indicators = get_technical_indicators(df)
        summary['technical_indicators'] = indicators
        
        return summary
    
    def prepare_model_data(self, df, target_column=None):
        """모델 학습을 위한 데이터 준비"""
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Unnamed: 0', 'index']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 타겟 컬럼이 지정되지 않은 경우 첫 번째 가격 컬럼 사용
        if target_column is None:
            price_cols = [col for col in feature_cols if any(x in col.lower() for x in ['btc', 'price'])]
            target_column = price_cols[0] if price_cols else feature_cols[0]
        
        # 타겟 컬럼이 피처에 포함되어 있으면 제거
        if target_column in feature_cols:
            feature_cols.remove(target_column)
        
        # 결측값 처리
        df_clean = df[feature_cols + [target_column]].dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[target_column]
        
        print(f"모델 데이터 준비 완료:")
        print(f"  피처 수: {len(feature_cols)}")
        print(f"  샘플 수: {len(X)}")
        print(f"  타겟 컬럼: {target_column}")
        
        return X, y, feature_cols
    
    def create_lagged_features(self, df, target_column, lags=[1, 2, 3, 5, 10]):
        """시계열 특성을 위한 지연 피처 생성"""
        df_lagged = df.copy()
        
        for lag in lags:
            df_lagged[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # 결측값 제거
        df_lagged = df_lagged.dropna()
        
        print(f"지연 피처 생성 완료: {len(lags)}개 지연 피처 추가")
        return df_lagged
    
    def split_time_series_data(self, df, target_column, test_size=0.2):
        """시계열 데이터 분할"""
        split_idx = int(len(df) * (1 - test_size))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        print(f"시계열 데이터 분할:")
        print(f"  훈련 데이터: {len(train_df)} 샘플")
        print(f"  테스트 데이터: {len(test_df)} 샘플")
        
        return train_df, test_df 