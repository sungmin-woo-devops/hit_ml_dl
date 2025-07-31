import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import warnings
warnings.filterwarnings('ignore')

print("뇌졸중 예측 데이터 ydata-profiling 분석")
print("=" * 50)

# 데이터 로드
print("데이터 로드 중...")
df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

# 데이터 기본 정보 확인
print(f"원본 데이터 shape: {df.shape}")
print("\n원본 데이터 구조:")
print(df.head())
print("\n컬럼 정보:")
print(df.columns.tolist())
print("\n데이터 타입:")
print(df.dtypes)

# 데이터 전처리
print("\n데이터 전처리 중...")

# 결측치 처리
print("결측치 정보:")
missing_info = df.isnull().sum()
missing_cols = missing_info[missing_info > 0]
if len(missing_cols) > 0:
    print(missing_cols)
else:
    print("결측값이 없습니다.")

# BMI 결측치 처리 (N/A 값)
df['bmi'] = df['bmi'].replace('N/A', np.nan)
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df['bmi'].fillna(df['bmi'].median(), inplace=True)

# 성별에서 'Other' 제거 (데이터가 너무 적음)
df = df[df['gender'] != 'Other']

# ID 컬럼 제거 (분석에 불필요)
df = df.drop('id', axis=1)

print(f"전처리 후 데이터 shape: {df.shape}")
print("\n전처리 후 데이터 정보:")
print(df.info())

print("\n기본 통계:")
print(df.describe())

# 추가 파생 변수 생성
print("\n파생 변수 생성 중...")

# 연령대 분류
df['age_group'] = pd.cut(df['age'], 
                         bins=[0, 20, 40, 60, 80, 100], 
                         labels=['20세 이하', '21-40세', '41-60세', '61-80세', '80세 이상'])

# BMI 분류
df['bmi_category'] = pd.cut(df['bmi'], 
                           bins=[0, 18.5, 25, 30, 50], 
                           labels=['저체중', '정상', '과체중', '비만'])

# 혈당 수준 분류
df['glucose_level'] = pd.cut(df['avg_glucose_level'], 
                            bins=[0, 100, 126, 300], 
                            labels=['정상', '전단계 당뇨', '당뇨'])

# 위험 요인 점수 계산
df['risk_score'] = 0
df.loc[df['age'] >= 65, 'risk_score'] += 2
df.loc[df['age'] >= 45, 'risk_score'] += 1
df.loc[df['hypertension'] == 1, 'risk_score'] += 2
df.loc[df['heart_disease'] == 1, 'risk_score'] += 2
df.loc[df['bmi'] >= 30, 'risk_score'] += 1
df.loc[df['avg_glucose_level'] >= 126, 'risk_score'] += 1
df.loc[df['smoking_status'].isin(['smokes', 'formerly smoked']), 'risk_score'] += 1

# 위험도 분류
df['risk_level'] = pd.cut(df['risk_score'], 
                          bins=[0, 2, 4, 6, 10], 
                          labels=['낮음', '보통', '높음', '매우 높음'])

# 연령과 성별 조합
df['age_gender'] = df['age_group'].astype(str) + '_' + df['gender']

# 거주지와 직업 조합
df['residence_work'] = df['Residence_type'] + '_' + df['work_type']

# 결혼 상태와 성별 조합
df['marital_gender'] = df['ever_married'] + '_' + df['gender']

# 흡연 상태와 성별 조합
df['smoking_gender'] = df['smoking_status'] + '_' + df['gender']

# 고혈압과 심장병 조합
df['cardio_condition'] = df['hypertension'].astype(str) + '_' + df['heart_disease'].astype(str)

# BMI와 혈당 조합
df['bmi_glucose'] = df['bmi_category'].astype(str) + '_' + df['glucose_level'].astype(str)

# 연령대별 위험도
df['age_risk'] = df['age_group'].astype(str) + '_' + df['risk_level'].astype(str)

# 직업별 위험도
df['work_risk'] = df['work_type'] + '_' + df['risk_level'].astype(str)

# 거주지별 위험도
df['residence_risk'] = df['Residence_type'] + '_' + df['risk_level'].astype(str)

# 성별별 위험도
df['gender_risk'] = df['gender'] + '_' + df['risk_level'].astype(str)

# 흡연 상태별 위험도
df['smoking_risk'] = df['smoking_status'] + '_' + df['risk_level'].astype(str)

# 결혼 상태별 위험도
df['marital_risk'] = df['ever_married'] + '_' + df['risk_level'].astype(str)

# 고혈압별 위험도
df['hypertension_risk'] = df['hypertension'].astype(str) + '_' + df['risk_level'].astype(str)

# 심장병별 위험도
df['heart_disease_risk'] = df['heart_disease'].astype(str) + '_' + df['risk_level'].astype(str)

# BMI별 위험도
df['bmi_risk'] = df['bmi_category'].astype(str) + '_' + df['risk_level'].astype(str)

# 혈당별 위험도
df['glucose_risk'] = df['glucose_level'].astype(str) + '_' + df['risk_level'].astype(str)

print(f"파생 변수 추가 후 shape: {df.shape}")
print(f"총 컬럼 수: {len(df.columns)}")

# 결측값 정보
print(f"\n결측값 정보:")
missing_info = df.isnull().sum()
missing_cols = missing_info[missing_info > 0]
if len(missing_cols) > 0:
    print(missing_cols)
else:
    print("결측값이 없습니다.")

# 뇌졸중 발생률 정보
print(f"\n뇌졸중 발생률:")
stroke_rate = df['stroke'].mean()
stroke_counts = df['stroke'].value_counts()
print(f"전체 뇌졸중 발생률: {stroke_rate:.2%}")
print(f"뇌졸중 발생 수: {stroke_counts[1]}명")
print(f"뇌졸중 미발생 수: {stroke_counts[0]}명")

# ydata-profiling 레포트 생성
print("\nydata-profiling 레포트 생성 중...")
print("이 과정은 데이터 크기에 따라 몇 분 정도 소요될 수 있습니다...")

# ProfileReport 설정
profile = ProfileReport(
    df,
    title="뇌졸중 예측 데이터 분석 레포트",
    dataset={
        "description": "뇌졸중 위험 요인을 분석하고 예측 모델의 성능을 평가하기 위한 의료 데이터셋",
        "copyright_holder": "Healthcare Analysis Team",
        "copyright_year": "2025"
    },
    variables={
        "descriptions": {
            "gender": "성별 (Male/Female)",
            "age": "나이",
            "hypertension": "고혈압 여부 (0/1)",
            "heart_disease": "심장병 여부 (0/1)",
            "ever_married": "결혼 여부 (Yes/No)",
            "work_type": "직업 유형 (Private/Self-employed/Govt_job/children/Never_worked)",
            "Residence_type": "거주 유형 (Urban/Rural)",
            "avg_glucose_level": "평균 혈당 수치",
            "bmi": "체질량지수 (BMI)",
            "smoking_status": "흡연 상태 (formerly smoked/never smoked/smokes/Unknown)",
            "stroke": "뇌졸중 발생 여부 (0/1)",
            "age_group": "연령대 분류",
            "bmi_category": "BMI 분류 (저체중/정상/과체중/비만)",
            "glucose_level": "혈당 수준 분류 (정상/전단계 당뇨/당뇨)",
            "risk_score": "위험 요인 점수",
            "risk_level": "위험도 분류 (낮음/보통/높음/매우 높음)",
            "age_gender": "연령대와 성별 조합",
            "residence_work": "거주지와 직업 조합",
            "marital_gender": "결혼 상태와 성별 조합",
            "smoking_gender": "흡연 상태와 성별 조합",
            "cardio_condition": "고혈압과 심장병 조합",
            "bmi_glucose": "BMI와 혈당 조합",
            "age_risk": "연령대별 위험도",
            "work_risk": "직업별 위험도",
            "residence_risk": "거주지별 위험도",
            "gender_risk": "성별별 위험도",
            "smoking_risk": "흡연 상태별 위험도",
            "marital_risk": "결혼 상태별 위험도",
            "hypertension_risk": "고혈압별 위험도",
            "heart_disease_risk": "심장병별 위험도",
            "bmi_risk": "BMI별 위험도",
            "glucose_risk": "혈당별 위험도"
        }
    },
    # 성능 최적화 설정
    minimal=False,
    explorative=True
)

# 레포트를 HTML 파일로 저장
print("HTML 레포트 저장 중...")
profile.to_file("stroke_profiling_report.html")

# 레포트를 JSON 파일로도 저장
print("JSON 레포트 저장 중...")
profile.to_file("stroke_profiling_report.json")

# 요약 통계 저장
print("요약 통계 저장 중...")
summary_stats = df.describe(include='all')
summary_stats.to_csv('stroke_summary_statistics.csv', encoding='utf-8-sig')

# 상관관계 행렬 저장
print("상관관계 분석 저장 중...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
correlation_matrix.to_csv('stroke_correlation_matrix.csv', encoding='utf-8-sig')

# 전처리된 데이터 저장
print("전처리된 데이터 저장 중...")
df.to_csv('stroke_processed_data.csv', index=False, encoding='utf-8-sig')

# 뇌졸중 발생률 분석
stroke_analysis = df.groupby('stroke').agg({
    'age': ['mean', 'std', 'min', 'max'],
    'bmi': ['mean', 'std'],
    'avg_glucose_level': ['mean', 'std'],
    'risk_score': ['mean', 'std']
}).round(4)

# 성별별 뇌졸중 발생률
gender_stroke = df.groupby('gender')['stroke'].agg(['count', 'sum', 'mean']).round(4)
gender_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 연령대별 뇌졸중 발생률
age_stroke = df.groupby('age_group')['stroke'].agg(['count', 'sum', 'mean']).round(4)
age_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 직업별 뇌졸중 발생률
work_stroke = df.groupby('work_type')['stroke'].agg(['count', 'sum', 'mean']).round(4)
work_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 거주지별 뇌졸중 발생률
residence_stroke = df.groupby('Residence_type')['stroke'].agg(['count', 'sum', 'mean']).round(4)
residence_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 흡연 상태별 뇌졸중 발생률
smoking_stroke = df.groupby('smoking_status')['stroke'].agg(['count', 'sum', 'mean']).round(4)
smoking_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 결혼 상태별 뇌졸중 발생률
marital_stroke = df.groupby('ever_married')['stroke'].agg(['count', 'sum', 'mean']).round(4)
marital_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 고혈압별 뇌졸중 발생률
hypertension_stroke = df.groupby('hypertension')['stroke'].agg(['count', 'sum', 'mean']).round(4)
hypertension_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 심장병별 뇌졸중 발생률
heart_disease_stroke = df.groupby('heart_disease')['stroke'].agg(['count', 'sum', 'mean']).round(4)
heart_disease_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# BMI 분류별 뇌졸중 발생률
bmi_stroke = df.groupby('bmi_category')['stroke'].agg(['count', 'sum', 'mean']).round(4)
bmi_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 혈당 수준별 뇌졸중 발생률
glucose_stroke = df.groupby('glucose_level')['stroke'].agg(['count', 'sum', 'mean']).round(4)
glucose_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 위험도별 뇌졸중 발생률
risk_stroke = df.groupby('risk_level')['stroke'].agg(['count', 'sum', 'mean']).round(4)
risk_stroke.columns = ['총 인원', '뇌졸중 발생 수', '뇌졸중 발생률']

# 통계 저장
stroke_analysis.to_csv('stroke_analysis.csv', encoding='utf-8-sig')
gender_stroke.to_csv('stroke_gender_analysis.csv', encoding='utf-8-sig')
age_stroke.to_csv('stroke_age_analysis.csv', encoding='utf-8-sig')
work_stroke.to_csv('stroke_work_analysis.csv', encoding='utf-8-sig')
residence_stroke.to_csv('stroke_residence_analysis.csv', encoding='utf-8-sig')
smoking_stroke.to_csv('stroke_smoking_analysis.csv', encoding='utf-8-sig')
marital_stroke.to_csv('stroke_marital_analysis.csv', encoding='utf-8-sig')
hypertension_stroke.to_csv('stroke_hypertension_analysis.csv', encoding='utf-8-sig')
heart_disease_stroke.to_csv('stroke_heart_disease_analysis.csv', encoding='utf-8-sig')
bmi_stroke.to_csv('stroke_bmi_analysis.csv', encoding='utf-8-sig')
glucose_stroke.to_csv('stroke_glucose_analysis.csv', encoding='utf-8-sig')
risk_stroke.to_csv('stroke_risk_analysis.csv', encoding='utf-8-sig')

# 분석 결과 요약
print("\n" + "=" * 80)
print("🏥 뇌졸중 예측 데이터 ydata-profiling 분석 완료!")
print("=" * 80)

print(f"\n📊 데이터 개요:")
print(f"• 총 샘플 수: {len(df):,}명")
print(f"• 뇌졸중 발생 수: {df['stroke'].sum():,}명")
print(f"• 뇌졸중 발생률: {df['stroke'].mean():.2%}")
print(f"• 총 변수 수: {len(df.columns)}개")
print(f"• 수치형 변수: {len(df.select_dtypes(include=[np.number]).columns)}개")
print(f"• 범주형 변수: {len(df.select_dtypes(include=['object', 'category']).columns)}개")

print(f"\n👥 인구통계학적 특성:")
print(f"• 평균 나이: {df['age'].mean():.1f}세")
print(f"• 남성 비율: {(df['gender'] == 'Male').mean():.1%}")
print(f"• 여성 비율: {(df['gender'] == 'Female').mean():.1%}")
print(f"• 고혈압 비율: {df['hypertension'].mean():.1%}")
print(f"• 심장병 비율: {df['heart_disease'].mean():.1%}")

print(f"\n📈 건강 지표:")
print(f"• 평균 BMI: {df['bmi'].mean():.1f}")
print(f"• 평균 혈당: {df['avg_glucose_level'].mean():.1f}")
print(f"• 평균 위험 점수: {df['risk_score'].mean():.1f}")

print(f"\n🏠 사회경제적 특성:")
print(f"• 결혼한 사람 비율: {(df['ever_married'] == 'Yes').mean():.1%}")
print(f"• 도시 거주 비율: {(df['Residence_type'] == 'Urban').mean():.1%}")
print(f"• 사설업체 종사자 비율: {(df['work_type'] == 'Private').mean():.1%}")

print(f"\n🚬 생활습관:")
smoking_counts = df['smoking_status'].value_counts()
for status, count in smoking_counts.items():
    print(f"• {status}: {count:,}명 ({count/len(df):.1%})")

print(f"\n🎯 주요 위험 요인 분석:")
print(f"• 고혈압 환자의 뇌졸중 발생률: {df[df['hypertension']==1]['stroke'].mean():.2%}")
print(f"• 심장병 환자의 뇌졸중 발생률: {df[df['heart_disease']==1]['stroke'].mean():.2%}")
print(f"• 65세 이상의 뇌졸중 발생률: {df[df['age']>=65]['stroke'].mean():.2%}")
print(f"• 비만 환자의 뇌졸중 발생률: {df[df['bmi']>=30]['stroke'].mean():.2%}")

print(f"\n📊 연령대별 뇌졸중 발생률:")
age_stroke_sorted = age_stroke.sort_values('뇌졸중 발생률', ascending=False)
for age_group, row in age_stroke_sorted.iterrows():
    print(f"• {age_group}: {row['뇌졸중 발생률']:.2%} ({row['뇌졸중 발생 수']}명)")

print(f"\n👨‍💼 직업별 뇌졸중 발생률:")
work_stroke_sorted = work_stroke.sort_values('뇌졸중 발생률', ascending=False)
for work_type, row in work_stroke_sorted.iterrows():
    print(f"• {work_type}: {row['뇌졸중 발생률']:.2%} ({row['뇌졸중 발생 수']}명)")

print(f"\n🚬 흡연 상태별 뇌졸중 발생률:")
smoking_stroke_sorted = smoking_stroke.sort_values('뇌졸중 발생률', ascending=False)
for smoking_status, row in smoking_stroke_sorted.iterrows():
    print(f"• {smoking_status}: {row['뇌졸중 발생률']:.2%} ({row['뇌졸중 발생 수']}명)")

print(f"\n📁 생성된 파일:")
print("• stroke_profiling_report.html - 종합 프로파일링 레포트 (메인)")
print("• stroke_profiling_report.json - JSON 형태 레포트")
print("• stroke_summary_statistics.csv - 요약 통계")
print("• stroke_correlation_matrix.csv - 상관관계 행렬")
print("• stroke_processed_data.csv - 전처리된 데이터")
print("• stroke_analysis.csv - 뇌졸중 분석 통계")
print("• stroke_gender_analysis.csv - 성별 분석")
print("• stroke_age_analysis.csv - 연령대별 분석")
print("• stroke_work_analysis.csv - 직업별 분석")
print("• stroke_residence_analysis.csv - 거주지별 분석")
print("• stroke_smoking_analysis.csv - 흡연 상태별 분석")
print("• stroke_marital_analysis.csv - 결혼 상태별 분석")
print("• stroke_hypertension_analysis.csv - 고혈압별 분석")
print("• stroke_heart_disease_analysis.csv - 심장병별 분석")
print("• stroke_bmi_analysis.csv - BMI별 분석")
print("• stroke_glucose_analysis.csv - 혈당별 분석")
print("• stroke_risk_analysis.csv - 위험도별 분석")

print(f"\n💡 분석 인사이트:")
print("• HTML 레포트에서 다음을 확인할 수 있습니다:")
print("  - 각 변수의 상세 분포 및 히스토그램")
print("  - 뇌졸중과 각 변수의 관계 분석")
print("  - 위험 요인별 뇌졸중 발생률 비교")
print("  - 변수 간 상관관계 분석")
print("  - 이상치 및 결측값 분석")
print("  - 연령대별, 성별, 직업별 패턴 분석")
print("  - 건강 지표와 뇌졸중의 관계")
print("  - 생활습관과 뇌졸중의 관계")

print(f"\n✅ ydata-profiling 분석 완료!")
print("웹브라우저에서 'stroke_profiling_report.html'을 열어 상세 레포트를 확인하세요!") 