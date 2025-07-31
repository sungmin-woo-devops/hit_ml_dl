import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
from imblearn.over_sampling import SMOTE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
import joblib
import os

# 폰트지정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.float_format = '{:.2f}'.format

# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 분석 시스템",
    page_icon="🏥",
    layout="wide"
)

# 제목
st.title("🏥 뇌졸중 예측 분석 시스템")
st.write("뇌졸중 위험 요인을 분석하고 예측 모델의 성능을 평가합니다.")

# 데이터 로드 함수
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../dataset/healthcare-dataset-stroke-data.csv", encoding='utf-8')
        return df
    except FileNotFoundError:
        st.error("데이터 파일을 찾을 수 없습니다. '../dataset/healthcare-dataset-stroke-data.csv' 파일이 필요합니다.")
        return None

# 데이터 전처리 함수
def preprocess_data(df):
    # 결측치 처리
    df['bmi'].fillna(df['bmi'].median(), inplace=True)
    df = df[df['gender'] != 'Other'].drop(columns='id')
    
    return df

# 모델 학습 함수
@st.cache_data
def train_model(df):
    # 특성 분류
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features.remove('stroke')
    
    # 데이터 분할
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    # 전처리 구성
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )
    
    # 학습/데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    # 전처리 실행
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # SMOTE를 사용한 최적화된 데이터셋 생성
    X_train_resampled, y_train_resampled, _ = create_optimized_dataset(
        X_train, y_train, X_test, y_test, preprocessor
    )
    
    # 모델 학습
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_resampled, y_train_resampled)
    
    # 예측 및 평가
    y_pred = model.predict(X_test_preprocessed)
    
    # 특성 중요도 계산
    categorical_feature_names = []
    for i, col in enumerate(categorical_features):
        categories = preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[i]
        categorical_feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    all_feature_names = numeric_features + categorical_feature_names
    
    feature_importance = pd.DataFrame({
        'feature': all_feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, preprocessor, X_test, y_test, y_pred, feature_importance

# Optuna 하이퍼파라미터 최적화 함수
def optimize_hyperparameters(X_train, y_train, X_test, y_test, preprocessor):
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        # SMOTE를 사용한 최적화된 데이터셋 생성
        X_train_resampled, y_train_resampled, X_test_preprocessed = create_optimized_dataset(
            X_train, y_train, X_test, y_test, preprocessor
        )
        
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_preprocessed)
        
        return f1_score(y_test, y_pred, average='weighted')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    return study.best_params, study.best_value, study

# 통계 지표 계산 함수
def calculate_statistics(y_true, y_pred, y_pred_proba):
    """추가 통계 지표 계산"""
    from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Balanced Accuracy
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    
    return mcc, kappa, bal_acc

# 학습 곡선 계산 함수
def plot_learning_curve(model, X, y, title="학습 곡선"):
    """학습 곡선 계산 및 시각화"""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1_weighted'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='훈련 점수',
        line=dict(color='blue'),
        error_y=dict(type='data', array=train_std, visible=True)
    ))
    
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='검증 점수',
        line=dict(color='red'),
        error_y=dict(type='data', array=val_std, visible=True)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='훈련 샘플 수',
        yaxis_title='F1 Score',
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

# ROC 커브 계산 함수
def plot_roc_curve(y_true, y_pred_proba, title="ROC 커브"):
    """ROC 커브 계산 및 시각화"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC 커브 (AUC = {roc_auc:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='무작위 분류기',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    return fig, roc_auc

# 상관관계 Heatmap 계산 함수
def plot_correlation_heatmap(df, title="상관관계 Heatmap"):
    """상관관계 Heatmap 계산 및 시각화"""
    # 수치형 변수만 선택
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    
    # 상관관계 계산
    correlation_matrix = numeric_df.corr()
    reversed_y = correlation_matrix.columns[::-1]
    reversed_z = correlation_matrix.values[::-1]

    # Heatmap 생성
    fig = go.Figure(data=go.Heatmap(
        z=reversed_z,
        x=correlation_matrix.columns,
        y=reversed_y,
        colorscale=px.colors.diverging.Cool,
        zmid=0,
        text=np.round(reversed_z, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="변수",
        yaxis_title="변수",
        width=600,
        height=500
    )
    
    return fig

# 최적화된 모델 학습 및 저장 함수
def train_and_save_optimized_model(df, best_params):
    # 특성 분류
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features.remove('stroke')
    
    # 데이터 분할
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    
    # 전처리 구성
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough"
    )
    
    # 학습/데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    
    # 전처리 실행
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # SMOTE를 사용한 최적화된 데이터셋 생성
    X_train_resampled, y_train_resampled, _ = create_optimized_dataset(
        X_train, y_train, X_test, y_test, preprocessor
    )
    
    # 최적화된 하이퍼파라미터로 모델 학습
    optimized_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    
    optimized_model.fit(X_train_resampled, y_train_resampled)
    
    # 모델과 전처리기 저장
    model_data = {
        'model': optimized_model,
        'preprocessor': preprocessor,
        'feature_names': X.columns.tolist(),
        'best_params': best_params
    }
    
    # models 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    # 모델 저장
    joblib.dump(model_data, 'models/stroke_prediction_model.pkl')
    
    # 예측 및 평가
    y_pred = optimized_model.predict(X_test_preprocessed)
    
    return optimized_model, preprocessor, X_test, y_test, y_pred, best_params

# 저장된 모델 로드 함수
def load_saved_model():
    try:
        model_data = joblib.load('models/stroke_prediction_model.pkl')
        return model_data
    except FileNotFoundError:
        return None

# SMOTE 최적화 함수 추가
def create_optimized_dataset(X_train, y_train, X_test, y_test, preprocessor, smote_params=None):
    """
    SMOTE를 사용하여 최적화된 데이터셋을 생성하는 함수
    
    Args:
        X_train, y_train: 훈련 데이터
        X_test, y_test: 테스트 데이터  
        preprocessor: 전처리기
        smote_params: SMOTE 파라미터 (기본값: random_state=42)
    
    Returns:
        X_train_resampled, y_train_resampled: 리샘플링된 훈련 데이터
        X_test_preprocessed: 전처리된 테스트 데이터
    """
    if smote_params is None:
        smote_params = {'random_state': 42}
    
    # 데이터 전처리
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # SMOTE 적용
    smote = SMOTE(**smote_params)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_preprocessed, y_train)
    
    return X_train_resampled, y_train_resampled, X_test_preprocessed

# SMOTE 파라미터 최적화 함수
def optimize_smote_parameters(X_train, y_train, X_test, y_test, preprocessor):
    """
    SMOTE 파라미터를 최적화하는 함수
    
    Returns:
        best_smote_params: 최적화된 SMOTE 파라미터
    """
    def objective(trial):
        # SMOTE 파라미터 제안
        k_neighbors = trial.suggest_int('k_neighbors', 3, 10)
        sampling_strategy = trial.suggest_float('sampling_strategy', 0.5, 1.0)
        
        smote_params = {
            'k_neighbors': k_neighbors,
            'sampling_strategy': sampling_strategy,
            'random_state': 42
        }
        
        # 최적화된 데이터셋 생성
        X_train_resampled, y_train_resampled, X_test_preprocessed = create_optimized_dataset(
            X_train, y_train, X_test, y_test, preprocessor, smote_params
        )
        
        # 간단한 모델로 평가
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_preprocessed)
        
        return f1_score(y_test, y_pred, average='weighted')
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    return study.best_params

# 메인 애플리케이션
def main():
    # 데이터 로드
    with st.spinner('데이터를 불러오는 중...'):
        df = load_data()
    
    if df is None:
        return
    
    # 데이터 전처리
    with st.spinner('데이터 전처리 중...'):
        df = preprocess_data(df)
    
    # 기본 정보 표시
    st.subheader("📊 데이터 기본 정보")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("총 샘플 수", f"{len(df):,}")
    with col2:
        st.metric("뇌졸중 환자 수", f"{df['stroke'].sum():,}")
    with col3:
        st.metric("뇌졸중 비율", f"{df['stroke'].mean():.2%}")
    with col4:
        st.metric("특성 수", f"{len(df.columns)-1}")
    
    # 데이터 미리보기
    st.subheader("📋 데이터 미리보기")
    st.dataframe(df.head(), use_container_width=True)
    
    # 데이터 정보
    with st.expander("데이터 상세 정보"):
        st.write("**데이터 타입:**")
        st.write(df.dtypes)
        
        st.write("**기술 통계:**")
        st.write(df.describe())
        
        st.write("**결측치 정보:**")
        st.write(df.isnull().sum())
    
    # 모델 학습
    with st.spinner('모델 학습 중...'):
        model, preprocessor, X_test, y_test, y_pred, feature_importance = train_model(df)
        
        # 학습/테스트 데이터 분할 정보 저장 (학습 곡선용)
        X = df.drop("stroke", axis=1)
        y = df["stroke"]
        X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        
        # 테스트 데이터 전처리 (ROC 커브용)
        X_test_preprocessed = preprocessor.transform(X_test)
    
    # 저장된 모델 확인
    saved_model = load_saved_model()
    
    # Optuna 하이퍼파라미터 최적화 및 모델 저장
    optimization_option = st.selectbox(
        "최적화 옵션 선택",
        ["기본 모델 사용", "하이퍼파라미터 최적화", "SMOTE 파라미터 최적화", "하이퍼파라미터 + SMOTE 최적화"]
    )
    
    if optimization_option == "하이퍼파라미터 최적화":
        with st.spinner('Optuna로 하이퍼파라미터 최적화 중...'):
            X_train, X_test_opt, y_train, y_test_opt = train_test_split(
                df.drop("stroke", axis=1), df["stroke"], 
                stratify=df["stroke"], test_size=0.2, random_state=42
            )
            best_params, best_score, study = optimize_hyperparameters(X_train, y_train, X_test_opt, y_test_opt, preprocessor)
    
    elif optimization_option == "SMOTE 파라미터 최적화":
        with st.spinner('SMOTE 파라미터 최적화 중...'):
            X_train, X_test_opt, y_train, y_test_opt = train_test_split(
                df.drop("stroke", axis=1), df["stroke"], 
                stratify=df["stroke"], test_size=0.2, random_state=42
            )
            best_smote_params = optimize_smote_parameters(X_train, y_train, X_test_opt, y_test_opt, preprocessor)
            st.success("SMOTE 파라미터 최적화 완료!")
            st.write("**최적 SMOTE 파라미터:**")
            st.json(best_smote_params)
    
    elif optimization_option == "하이퍼파라미터 + SMOTE 최적화":
        with st.spinner('하이퍼파라미터와 SMOTE 파라미터 동시 최적화 중...'):
            X_train, X_test_opt, y_train, y_test_opt = train_test_split(
                df.drop("stroke", axis=1), df["stroke"], 
                stratify=df["stroke"], test_size=0.2, random_state=42
            )
            best_params, best_score, study = optimize_hyperparameters(X_train, y_train, X_test_opt, y_test_opt, preprocessor)
            best_smote_params = optimize_smote_parameters(X_train, y_train, X_test_opt, y_test_opt, preprocessor)
            st.success("모든 파라미터 최적화 완료!")
            st.write("**최적 하이퍼파라미터:**")
            st.json(best_params)
            st.write("**최적 SMOTE 파라미터:**")
            st.json(best_smote_params)
    
    # 최적화 결과 표시 및 모델 저장
    if optimization_option in ["하이퍼파라미터 최적화", "하이퍼파라미터 + SMOTE 최적화"]:
        st.success(f"최적화 완료! 최고 F1-Score: {best_score:.4f}")
        st.write("**최적 하이퍼파라미터:**")
        st.json(best_params)
        
        # 최적화된 모델 학습 및 저장
        with st.spinner('최적화된 모델 학습 및 저장 중...'):
            optimized_model, optimized_preprocessor, X_test_opt, y_test_opt, y_pred_opt, best_params = train_and_save_optimized_model(df, best_params)
            
            st.success("✅ 최적화된 모델이 'models/stroke_prediction_model.pkl'에 저장되었습니다!")
            
            # 최적화된 모델 성능 표시
            report_opt = classification_report(y_test_opt, y_pred_opt, output_dict=True)
            st.write("**최적화된 모델 성능:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("정확도", f"{report_opt['accuracy']:.3f}")
            with col2:
                st.metric("정밀도", f"{report_opt['1']['precision']:.3f}")
            with col3:
                st.metric("재현율", f"{report_opt['1']['recall']:.3f}")
            with col4:
                st.metric("F1-Score", f"{report_opt['1']['f1-score']:.3f}")
    
    # 저장된 모델 사용
    elif saved_model is not None:
        st.success("✅ 저장된 최적화 모델을 사용합니다.")
        st.write("**저장된 모델 정보:**")
        st.json(saved_model['best_params'])
        
        # 저장된 모델로 예측 기능 추가
        st.subheader("🎯 개인 뇌졸중 위험 예측")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("나이", min_value=0, max_value=120, value=50)
                gender = st.selectbox("성별", ["Male", "Female"])
                hypertension = st.selectbox("고혈압", [0, 1])
                heart_disease = st.selectbox("심장병", [0, 1])
                ever_married = st.selectbox("결혼 여부", ["Yes", "No"])
            
            with col2:
                work_type = st.selectbox("직업 유형", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
                residence_type = st.selectbox("거주 유형", ["Urban", "Rural"])
                avg_glucose_level = st.number_input("평균 혈당 수치", min_value=50.0, max_value=300.0, value=100.0)
                bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
                smoking_status = st.selectbox("흡연 상태", ["formerly smoked", "never smoked", "smokes", "Unknown"])
            
            submitted = st.form_submit_button("예측하기")
            
            if submitted:
                # 입력 데이터 생성
                input_data = pd.DataFrame({
                    'age': [age],
                    'gender': [gender],
                    'hypertension': [hypertension],
                    'heart_disease': [heart_disease],
                    'ever_married': [ever_married],
                    'work_type': [work_type],
                    'Residence_type': [residence_type],
                    'avg_glucose_level': [avg_glucose_level],
                    'bmi': [bmi],
                    'smoking_status': [smoking_status]
                })
                
                # 전처리 및 예측
                input_preprocessed = saved_model['preprocessor'].transform(input_data)
                prediction = saved_model['model'].predict(input_preprocessed)
                prediction_proba = saved_model['model'].predict_proba(input_preprocessed)
                
                # 결과 표시
                if prediction[0] == 1:
                    st.error("⚠️ 뇌졸중 위험이 높습니다!")
                    st.write(f"뇌졸중 발생 확률: {prediction_proba[0][1]:.2%}")
                else:
                    st.success("✅ 뇌졸중 위험이 낮습니다.")
                    st.write(f"뇌졸중 발생 확률: {prediction_proba[0][1]:.2%}")
                
                # 위험도 시각화
                fig = px.bar(
                    x=['뇌졸중 없음', '뇌졸중 있음'],
                    y=prediction_proba[0],
                    title="뇌졸중 발생 확률",
                    labels={'x': '결과', 'y': '확률'}
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
    
    # 성능 지표 계산
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 성능 지표 표시
    st.subheader("🎯 모델 성능")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("정확도", f"{report['accuracy']:.3f}")
    with col2:
        st.metric("정밀도", f"{report['1']['precision']:.3f}")
    with col3:
        st.metric("재현율", f"{report['1']['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
    
    # 추가 통계 지표 계산
    y_pred_proba = model.predict_proba(X_test_preprocessed)
    mcc, kappa, bal_acc = calculate_statistics(y_test, y_pred, y_pred_proba)
    
    st.subheader("📊 추가 통계 지표")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Matthews Correlation", f"{mcc:.3f}")
    with col2:
        st.metric("Cohen's Kappa", f"{kappa:.3f}")
    with col3:
        st.metric("Balanced Accuracy", f"{bal_acc:.3f}")
    
    # 시각화 섹션
    st.subheader("📈 분석 결과 시각화")
    
    # 탭으로 구분
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "혼동 행렬", "특성 중요도", "인구통계 분석", "성능 지표", 
        "학습 곡선", "ROC 커브", "상관관계 Heatmap", "Optuna 최적화 과정"
    ])
    
    with tab1:
        # 혼동 행렬
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['뇌졸중 없음', '뇌졸중 있음'],
            y=['뇌졸중 없음', '뇌졸중 있음'],
            colorscale='Blues',
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 16}
        ))
        fig.update_layout(
            title="혼동 행렬 (Confusion Matrix)",
            xaxis_title="예측값",
            yaxis_title="실제값"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 특성 중요도
        top_features = feature_importance.head(10)
        fig = px.bar(
            top_features, 
            x='importance', 
            y='feature',
            orientation='h',
            title="상위 10개 특성 중요도"
        )
        fig.update_layout(xaxis_title="중요도", yaxis_title="특성")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # 인구통계 분석
        col1, col2 = st.columns(2)
        
        with col1:
            # 연령대별 뇌졸중 발생률
            df['age_group'] = pd.cut(df['age'], 
                                    bins=[0, 20, 40, 60, 80, 100], 
                                    labels=['0-20', '21-40', '41-60', '61-80', '80+'])
            age_stroke_rate = df.groupby('age_group')['stroke'].mean()
            
            fig = px.bar(
                x=age_stroke_rate.index,
                y=age_stroke_rate.values,
                title="연령대별 뇌졸중 발생률",
                labels={'x': '연령대', 'y': '뇌졸중 발생률'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 성별 뇌졸중 발생률
            gender_stroke_rate = df.groupby('gender')['stroke'].mean()
            fig = px.bar(
                x=gender_stroke_rate.index,
                y=gender_stroke_rate.values,
                title="성별 뇌졸중 발생률",
                labels={'x': '성별', 'y': '뇌졸중 발생률'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # 성능 지표 비교
        metrics = ['정확도', '정밀도', '재현율', 'F1-Score']
        values = [report['accuracy'], report['1']['precision'], 
                 report['1']['recall'], report['1']['f1-score']]
        
        fig = px.bar(
            x=metrics,
            y=values,
            title="모델 성능 지표 비교",
            labels={'x': '지표', 'y': '점수'}
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        # 학습 곡선
        with st.spinner('학습 곡선 계산 중...'):
            X_train_preprocessed = preprocessor.transform(X_train)
            learning_fig = plot_learning_curve(model, X_train_preprocessed, y_train)
            st.plotly_chart(learning_fig, use_container_width=True)
    
    with tab6:
        # ROC 커브
        with st.spinner('ROC 커브 계산 중...'):
            roc_fig, roc_auc = plot_roc_curve(y_test, y_pred_proba)
            st.plotly_chart(roc_fig, use_container_width=True)
            st.info(f"ROC AUC: {roc_auc:.3f}")
    
    with tab7:
        # 상관관계 Heatmap
        with st.spinner('상관관계 Heatmap 계산 중...'):
            correlation_fig = plot_correlation_heatmap(df)
            st.plotly_chart(
                correlation_fig, 
                use_container_width=True,
                config=dict(displayModeBar=False)
            )
            
            # 상관관계 설명 추가
            st.subheader("📊 상관관계 분석 설명")
            st.write("""
            **상관관계 해석:**
            - **1.0 (빨간색)**: 완벽한 양의 상관관계
            - **0.0 (흰색)**: 상관관계 없음
            - **-1.0 (파란색)**: 완벽한 음의 상관관계
            
            **주요 발견사항:**
            - 뇌졸중과 가장 높은 상관관계를 보이는 변수들을 확인할 수 있습니다
            - 변수 간 다중공선성을 파악할 수 있습니다
            """)
            
            # 뇌졸중과의 상관관계 순위 표시
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            stroke_corr = numeric_df.corr()['stroke'].sort_values(ascending=False)
            
            st.subheader("🎯 뇌졸중과의 상관관계 순위")
            corr_df = pd.DataFrame({
                '변수': stroke_corr.index,
                '상관계수': stroke_corr.values
            })
            st.dataframe(corr_df, use_container_width=True)
            
            # 상관관계가 높은 변수들 시각화
            high_corr_vars = stroke_corr[abs(stroke_corr) > 0.1].index.tolist()
            if len(high_corr_vars) > 1:  # stroke 제외하고 1개 이상
                st.subheader("📈 높은 상관관계 변수들")
                fig = px.bar(
                    x=high_corr_vars,
                    y=stroke_corr[high_corr_vars].values,
                    title="뇌졸중과 높은 상관관계를 보이는 변수들",
                    labels={'x': '변수', 'y': '상관계수'}
                )
                fig.update_layout(yaxis_range=[-1, 1])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab8:
        # Optuna 최적화 과정 (최적화가 실행된 경우에만 표시)
        if 'study' in globals():
            # 최적화 과정 시각화
            fig = go.Figure()
            
            # 최적화 과정의 점수 변화
            trials = study.trials
            scores = [trial.value for trial in trials if trial.value is not None]
            trial_numbers = list(range(1, len(scores) + 1))
            
            fig.add_trace(go.Scatter(
                x=trial_numbers,
                y=scores,
                mode='lines+markers',
                name='F1 Score',
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="Optuna 하이퍼파라미터 최적화 과정",
                xaxis_title="시도 횟수",
                yaxis_title="F1 Score",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 최적화 과정 상세 정보
            st.write("**최적화 과정 상세 정보:**")
            st.write(f"- 총 시도 횟수: {len(trials)}")
            st.write(f"- 최고 점수: {study.best_value:.4f}")
            st.write(f"- 최적 파라미터: {study.best_params}")
        else:
            st.info("하이퍼파라미터 최적화를 실행하면 최적화 과정을 확인할 수 있습니다.")
    
    # 상세 분석 결과
    st.subheader("📋 상세 분석 결과")
    
    with st.expander("분류 보고서"):
        st.text(classification_report(y_test, y_pred))
    
    with st.expander("특성 중요도 상세"):
        st.dataframe(feature_importance, use_container_width=True)
    
    # 데이터 분포 분석
    st.subheader("📊 데이터 분포 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 타겟 변수 분포
        stroke_counts = df['stroke'].value_counts()
        fig = px.pie(
            values=stroke_counts.values,
            names=['뇌졸중 없음', '뇌졸중 있음'],
            title="타겟 변수 분포"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 연령 분포
        fig = px.histogram(
            df, 
            x='age', 
            nbins=20,
            title="연령 분포"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 범주형 변수 분포
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        st.subheader("📊 범주형 변수 분포")
        
        for col in categorical_cols:
            fig = px.bar(
                x=df[col].value_counts().index,
                y=df[col].value_counts().values,
                title=f"{col} 분포",
                labels={'x': col, 'y': '빈도'}
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 