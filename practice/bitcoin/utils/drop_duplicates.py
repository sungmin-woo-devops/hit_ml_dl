import pandas as pd
import numpy as np
import re

def drop_duplicate_columns(csv_file, output_file=None):
    """중복 컬럼들을 제거하고 정리된 데이터를 저장하는 함수"""
    
    # CSV 파일 로드
    df = pd.read_csv(csv_file)
    
    print(f"원본 데이터 크기: {df.shape}")
    print(f"원본 컬럼 수: {len(df.columns)}")
    
    # 중복 컬럼 찾기
    duplicate_groups = []
    checked_columns = set()
    
    for i, col1 in enumerate(df.columns):
        if col1 in checked_columns:
            continue
            
        current_group = [col1]
        checked_columns.add(col1)
        
        for j, col2 in enumerate(df.columns[i+1:], i+1):
            if col2 in checked_columns:
                continue
                
            # 두 컬럼의 데이터가 동일한지 확인
            if df[col1].equals(df[col2]):
                current_group.append(col2)
                checked_columns.add(col2)
        
        if len(current_group) > 1:
            duplicate_groups.append(current_group)
    
    # 제거할 컬럼들 수집
    columns_to_drop = []
    keep_columns = []
    
    for group in duplicate_groups:
        # 첫 번째 컬럼은 유지하고 나머지는 제거
        keep_columns.append(group[0])
        columns_to_drop.extend(group[1:])
        print(f"중복 그룹 '{group[0]}' 패턴: {len(group)}개 컬럼 중 {group[0]} 유지, 나머지 제거")
    
    # 중복되지 않는 컬럼들도 유지
    for col in df.columns:
        if col not in checked_columns or col in keep_columns:
            if col not in keep_columns:
                keep_columns.append(col)
    
    # 중복 컬럼 제거
    df_cleaned = df[keep_columns].copy()
    
    print(f"\n정리된 데이터 크기: {df_cleaned.shape}")
    print(f"정리된 컬럼 수: {len(df_cleaned.columns)}")
    print(f"제거된 컬럼 수: {len(columns_to_drop)}")
    
    # 컬럼명 정리 (숫자 접미사 제거)
    column_mapping = {}
    for col in df_cleaned.columns:
        # 숫자 접미사 제거
        clean_name = re.sub(r'\.\d+$', '', col)
        if clean_name != col:
            column_mapping[col] = clean_name
            print(f"컬럼명 변경: {col} -> {clean_name}")
    
    # 컬럼명 변경
    df_cleaned = df_cleaned.rename(columns=column_mapping)
    
    # 최종 컬럼 목록 출력
    print(f"\n최종 컬럼 목록 ({len(df_cleaned.columns)}개):")
    for i, col in enumerate(df_cleaned.columns, 1):
        print(f"{i:2d}. {col}")
    
    # 파일 저장
    if output_file is None:
        output_file = csv_file.replace('.csv', '_cleaned.csv')
    
    df_cleaned.to_csv(output_file, index=False)
    print(f"\n정리된 데이터가 저장되었습니다: {output_file}")
    
    return df_cleaned

def analyze_cleaned_data(df):
    """정리된 데이터 분석"""
    print("\n" + "="*50)
    print("정리된 데이터 분석")
    print("="*50)
    
    # 데이터 타입별 컬럼 수
    dtype_counts = df.dtypes.value_counts()
    print(f"\n데이터 타입별 컬럼 수:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count}개")
    
    # 결측값 분석
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    
    if len(columns_with_missing) > 0:
        print(f"\n결측값이 있는 컬럼 ({len(columns_with_missing)}개):")
        for col, count in columns_with_missing.items():
            percentage = (count / len(df)) * 100
            print(f"  {col}: {count}개 ({percentage:.1f}%)")
    else:
        print("\n결측값이 있는 컬럼이 없습니다.")
    
    # 수치형 컬럼 통계
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\n수치형 컬럼 통계 ({len(numeric_cols)}개):")
        for col in numeric_cols:
            try:
                stats = df[col].describe()
                if 'mean' in stats.index:
                    print(f"  {col}:")
                    print(f"    평균: {stats['mean']:.4f}")
                    print(f"    표준편차: {stats['std']:.4f}")
                    print(f"    최소값: {stats['min']:.4f}")
                    print(f"    최대값: {stats['max']:.4f}")
                else:
                    print(f"  {col}: 통계 계산 불가")
            except Exception as e:
                print(f"  {col}: 오류 발생 - {e}")
    
    # 기술지표 컬럼 확인
    technical_indicators = [col for col in df.columns if any(x in col.lower() for x in ['rsi', 'bb_percent', 'ma_ratio', 'vol_ratio', 'log_returns'])]
    if technical_indicators:
        print(f"\n기술지표 컬럼 ({len(technical_indicators)}개):")
        for col in technical_indicators:
            non_null_count = df[col].notna().sum()
            print(f"  {col}: {non_null_count}개 유효 데이터")
    
    # 가격 관련 컬럼 확인
    price_columns = [col for col in df.columns if any(x in col.lower() for x in ['btc', 'usd', 'eur', 'jpy', 'krw', 'xau'])]
    if price_columns:
        print(f"\n가격 관련 컬럼 ({len(price_columns)}개):")
        for col in price_columns:
            print(f"  {col}")

if __name__ == "__main__":
    csv_file = "data/merged_data_features.csv"
    output_file = "data/merged_data_features_cleaned.csv"
    
    # 중복 컬럼 제거
    df_cleaned = drop_duplicate_columns(csv_file, output_file)
    
    # 정리된 데이터 분석
    analyze_cleaned_data(df_cleaned) 