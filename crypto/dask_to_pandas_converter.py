#!/usr/bin/env python3
"""
Dask DataFrame을 안전하게 Pandas DataFrame으로 변환하는 스크립트
메모리 효율적인 방법으로 대용량 데이터 처리
"""

import dask.dataframe as dd
import pandas as pd
import os
import gc
from typing import Optional, List

class DaskToPandasConverter:
    """Dask DataFrame을 Pandas DataFrame으로 안전하게 변환하는 클래스"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.train_df = None
        
    def load_dask_dataframe(self) -> dd.DataFrame:
        """Dask DataFrame 로드"""
        print("=== Dask DataFrame 로드 ===")
        
        # 메모리 효율적인 데이터 타입 설정
        dtype_dict = {
            'Asset_ID': 'int8', 
            'Close': 'float32', 
            'Open': 'float32', 
            'High': 'float32', 
            'Low': 'float32', 
            'Volume': 'float32', 
            'VWAP': 'float32', 
            'Target': 'float32'
        }
        
        train_file = os.path.join(self.data_dir, 'train.csv')
        self.train_df = dd.read_csv(train_file, dtype=dtype_dict)
        
        print(f"Dask DataFrame 정보:")
        print(f"  파티션 수: {self.train_df.npartitions}")
        print(f"  컬럼: {list(self.train_df.columns)}")
        print(f"  데이터 타입: {self.train_df.dtypes}")
        
        return self.train_df
    
    def get_sample_pandas(self, n_rows: int = 1000) -> pd.DataFrame:
        """샘플 데이터를 Pandas DataFrame으로 변환"""
        print(f"\n=== 처음 {n_rows}행을 Pandas DataFrame으로 변환 ===")
        
        if self.train_df is None:
            self.load_dask_dataframe()
        
        # 처음 n_rows만 변환
        df_sample = self.train_df.head(n_rows)
        
        print(f"샘플 데이터 정보:")
        print(f"  Shape: {df_sample.shape}")
        print(f"  메모리 사용량: {df_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df_sample
    
    def get_partition_samples(self, n_partitions: int = 3, rows_per_partition: int = 1000) -> pd.DataFrame:
        """각 파티션에서 샘플을 추출하여 결합"""
        print(f"\n=== {n_partitions}개 파티션에서 각각 {rows_per_partition}행씩 샘플링 ===")
        
        if self.train_df is None:
            self.load_dask_dataframe()
        
        partition_samples = []
        
        for i in range(min(n_partitions, self.train_df.npartitions)):
            print(f"파티션 {i+1} 처리 중...")
            partition = self.train_df.get_partition(i).head(rows_per_partition)
            partition_samples.append(partition)
        
        if partition_samples:
            combined_df = pd.concat(partition_samples, ignore_index=True)
            print(f"결합된 데이터 정보:")
            print(f"  Shape: {combined_df.shape}")
            print(f"  메모리 사용량: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return combined_df
        else:
            print("샘플링된 데이터가 없습니다.")
            return pd.DataFrame()
    
    def get_asset_sample(self, asset_id: int = 0, n_rows: int = 1000) -> pd.DataFrame:
        """특정 Asset_ID의 데이터만 샘플링"""
        print(f"\n=== Asset_ID {asset_id}의 {n_rows}행 샘플링 ===")
        
        if self.train_df is None:
            self.load_dask_dataframe()
        
        # 특정 Asset_ID 필터링 후 샘플링
        asset_data = self.train_df[self.train_df['Asset_ID'] == asset_id].head(n_rows)
        
        print(f"Asset_ID {asset_id} 데이터 정보:")
        print(f"  Shape: {asset_data.shape}")
        print(f"  메모리 사용량: {asset_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return asset_data
    
    def get_data_info(self, df: pd.DataFrame, name: str = "데이터프레임") -> None:
        """데이터프레임 정보 출력"""
        print(f"\n=== {name} 상세 정보 ===")
        print(f"Shape: {df.shape}")
        print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"컬럼 타입:")
        print(df.dtypes)
        print(f"\n처음 5행:")
        print(df.head())
        print(f"\n기본 통계:")
        print(df.describe())
        
        # 결측치 확인
        print(f"\n결측치 수:")
        print(df.isnull().sum())
    
    def cleanup_memory(self):
        """메모리 정리"""
        gc.collect()
        print("메모리 정리 완료")

def main():
    """메인 실행 함수"""
    # 데이터 디렉토리 설정
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    
    print(f"데이터 디렉토리: {data_dir}")
    
    # 변환기 초기화
    converter = DaskToPandasConverter(data_dir)
    
    # 1. Dask DataFrame 로드
    dask_df = converter.load_dask_dataframe()
    
    # 2. 다양한 방법으로 Pandas DataFrame 변환
    print("\n" + "="*50)
    
    # 방법 1: 처음 1000행만
    df_sample = converter.get_sample_pandas(1000)
    converter.get_data_info(df_sample, "샘플 데이터 (처음 1000행)")
    
    # 방법 2: 파티션별 샘플링
    df_partitions = converter.get_partition_samples(3, 1000)
    converter.get_data_info(df_partitions, "파티션 샘플링 데이터")
    
    # 방법 3: 특정 Asset_ID만
    df_asset_0 = converter.get_asset_sample(0, 1000)
    converter.get_data_info(df_asset_0, "Asset_ID=0 데이터")
    
    # 3. 메모리 정리
    converter.cleanup_memory()
    
    print("\n=== 변환 완료 ===")
    return df_sample, df_partitions, df_asset_0

if __name__ == "__main__":
    df_sample, df_partitions, df_asset_0 = main() 