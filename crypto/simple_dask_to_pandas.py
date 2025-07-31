# 노트북에서 바로 사용할 수 있는 Dask to Pandas 변환 코드

import dask.dataframe as dd
import pandas as pd
import gc

print("=== Dask DataFrame을 Pandas DataFrame으로 안전하게 변환 ===")

# 현재 Dask DataFrame이 이미 로드되어 있다고 가정 (train_df)
# 만약 없다면 다시 로드
if 'train_df' not in locals():
    print("Dask DataFrame을 다시 로드합니다...")
    train_df = dd.read_csv(DATA_DIR + 'train.csv', 
                           dtype={'Asset_ID': 'int8', 'Close': 'float32', 'Open': 'float32', 
                                  'High': 'float32', 'Low': 'float32', 'Volume': 'float32', 
                                  'VWAP': 'float32', 'Target': 'float32'})

# 방법 1: 처음 1000행만 변환 (가장 안전)
print("\n방법 1: 처음 1000행만 변환")
df_train_sample = train_df.head(1000)
print("df_train_sample.info():")
print(df_train_sample.info())
print("\ndf_train_sample.head():")
print(df_train_sample.head())

# 방법 2: 특정 Asset_ID만 필터링하여 변환
print("\n방법 2: Asset_ID=0 데이터만 샘플링")
asset_0_data = train_df[train_df['Asset_ID'] == 0].head(1000)
print("Asset_ID=0 데이터 샘플:")
print(asset_0_data.head())

# 방법 3: 각 파티션에서 샘플 추출
print("\n방법 3: 파티션별 샘플링")
partition_samples = []
for i in range(min(3, train_df.npartitions)):  # 처음 3개 파티션만
    partition = train_df.get_partition(i).head(500)  # 각 파티션에서 500행
    partition_samples.append(partition)

if partition_samples:
    df_train_combined = pd.concat(partition_samples, ignore_index=True)
    print("결합된 데이터 정보:")
    print(f"Shape: {df_train_combined.shape}")
    print(f"메모리 사용량: {df_train_combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n처음 5행:")
    print(df_train_combined.head())

# 방법 4: 특정 시간 구간만 샘플링
print("\n방법 4: 특정 시간 구간 샘플링")
# 예: timestamp가 특정 범위인 데이터만
time_filtered = train_df[train_df['timestamp'] >= 1514764860].head(1000)
print("시간 필터링된 데이터:")
print(time_filtered.head())

# 메모리 사용량 확인
print("\n=== 메모리 사용량 확인 ===")
if 'df_train_sample' in locals():
    print(f"샘플 데이터 메모리 사용량: {df_train_sample.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

if 'df_train_combined' in locals():
    print(f"결합된 데이터 메모리 사용량: {df_train_combined.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# 데이터 통계 정보
print("\n=== 데이터 통계 정보 ===")
if 'df_train_sample' in locals():
    print("샘플 데이터 통계:")
    print(df_train_sample.describe())
    
    print("\n결측치 확인:")
    print(df_train_sample.isnull().sum())

# 메모리 정리
gc.collect()
print("\n메모리 정리 완료")

print("\n=== 변환 완료 ===")
print("이제 df_train_sample, asset_0_data, df_train_combined 등을 사용할 수 있습니다.") 