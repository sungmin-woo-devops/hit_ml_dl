import pandas as pd
import numpy as np

def find_duplicate_columns(csv_file):
    """CSV 파일에서 중복되는 컬럼들을 찾는 함수"""
    
    # CSV 파일 로드
    df = pd.read_csv(csv_file)
    
    print(f"총 컬럼 수: {len(df.columns)}")
    print(f"총 행 수: {len(df)}")
    print("\n컬럼 목록:")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    print("\n" + "="*50)
    print("중복 컬럼 분석")
    print("="*50)
    
    # 컬럼별로 데이터 비교
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
    
    # 결과 출력
    if duplicate_groups:
        print(f"\n총 {len(duplicate_groups)}개의 중복 그룹을 발견했습니다:")
        
        for i, group in enumerate(duplicate_groups, 1):
            print(f"\n그룹 {i} ({len(group)}개 컬럼):")
            for col in group:
                print(f"  - {col}")
            
            # 첫 번째 컬럼의 샘플 데이터 출력
            sample_data = df[group[0]].head(5).tolist()
            print(f"  샘플 데이터: {sample_data}")
    else:
        print("\n중복되는 컬럼이 없습니다.")
    
    # 컬럼명 패턴 분석
    print("\n" + "="*50)
    print("컬럼명 패턴 분석")
    print("="*50)
    
    # 숫자 접미사가 있는 컬럼들 찾기
    import re
    
    pattern_groups = {}
    for col in df.columns:
        # 숫자 접미사 제거
        base_name = re.sub(r'\.\d+$', '', col)
        if base_name not in pattern_groups:
            pattern_groups[base_name] = []
        pattern_groups[base_name].append(col)
    
    # 패턴별로 그룹화
    for base_name, cols in pattern_groups.items():
        if len(cols) > 1:
            print(f"\n'{base_name}' 패턴 ({len(cols)}개):")
            for col in cols:
                print(f"  - {col}")
    
    return duplicate_groups

if __name__ == "__main__":
    csv_file = "data/merged_data_features.csv"
    duplicates = find_duplicate_columns(csv_file) 