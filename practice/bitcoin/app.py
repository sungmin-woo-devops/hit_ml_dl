"""
리팩토링된 비트코인 데이터 분석 대시보드
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# 모듈화된 bitcoin_modules import
from bitcoin_modules import (
    DataCollector, 
    FeatureEngineer, 
    DataProcessor, 
    ModelTrainer,
    setup_environment,
    get_data_paths
)

# 새로 생성한 모듈들 import
from bitcoin_modules.visualization import (
    create_price_chart,
    create_technical_indicators_chart,
    create_correlation_heatmap,
    create_prediction_scatter,
    create_time_series_prediction,
    create_residual_analysis,
    create_distribution_analysis,
    create_feature_importance_chart,
    create_learning_curve,
    create_confidence_analysis,
    create_optimization_process_chart,
    create_parameter_importance_chart
)

# sklearn 관련 import 추가
from sklearn.preprocessing import StandardScaler

from bitcoin_modules.ml_utils import (
    prepare_ml_data,
    train_random_forest_model,
    optimize_hyperparameters,
    get_feature_importance,
    calculate_ensemble_predictions,
    calculate_learning_curve,
    evaluate_model_performance,
    get_normality_test_results,
    calculate_statistical_measures,
    find_strong_correlations
)

from bitcoin_modules.ui_utils import (
    create_sidebar_settings,
    create_metrics_display,
    display_model_results,
    create_period_selector,
    get_period_days,
    display_technical_indicators_info,
    display_ml_model_info,
    create_data_download_section,
    display_system_info,
    display_ml_settings,
    create_progress_display,
    update_progress,
    clear_progress
)

# 경고 무시
warnings.filterwarnings('ignore')

# 환경 설정
setup_environment()

# 페이지 설정
st.set_page_config(
    page_title="비트코인 데이터 분석 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 경로 설정
paths = get_data_paths()
DATA_DIR = paths['data_dir']


@st.cache_data
def load_data():
    """데이터 로드 함수 - PKL 파일 우선 사용"""
    try:
        # 파일 경로 설정
        pkl_file_path = paths['pkl_data_path']
        cleaned_csv_path = paths['cleaned_data_path']
        original_csv_path = paths['after_feature_path']
        
        # PKL 파일 우선 시도
        if os.path.exists(pkl_file_path):
            df = pd.read_pickle(pkl_file_path)
            st.success(f"PKL 데이터 로드 성공: {pkl_file_path}")
            return df
        # CSV 파일 폴백
        elif os.path.exists(cleaned_csv_path):
            df = pd.read_csv(cleaned_csv_path)
            st.success(f"정리된 CSV 데이터 로드 성공: {cleaned_csv_path}")
            return df
        elif os.path.exists(original_csv_path):
            df = pd.read_csv(original_csv_path)
            st.success(f"원본 CSV 데이터 로드 성공: {original_csv_path}")
            return df
        else:
            st.error(f"데이터 파일을 찾을 수 없습니다. 다음 파일들을 확인해주세요:")
            st.error(f"- {pkl_file_path}")
            st.error(f"- {cleaned_csv_path}")
            st.error(f"- {original_csv_path}")
            return None
        
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        return None


def create_technical_indicators(df):
    """기술지표 생성 - 모듈화된 FeatureEngineer 사용"""
    if df is None:
        return df
    
    try:
        # FeatureEngineer 인스턴스 생성
        feature_engineer = FeatureEngineer()
        
        # 이미 기술지표가 있는지 확인
        existing_indicators = ['log_returns_x', 'rsi_x', 'bb_percent_x', 'ma_ratio_x', 'vol_ratio_x']
        has_indicators = any(col in df.columns for col in existing_indicators)
        
        if has_indicators:
            st.info("이미 기술지표가 포함된 데이터를 사용합니다.")
            return df
        
        # 기술지표 생성
        df_with_features = feature_engineer.create_all_features(df)
        
        return df_with_features
        
    except Exception as e:
        st.error(f"기술지표 생성 중 오류 발생: {e}")
        return df


def render_price_chart_tab(df, price_col, chart_style):
    """가격 차트 탭 렌더링"""
    st.header("가격 차트")
    price_chart = create_price_chart(df, price_col, chart_style)
    st.plotly_chart(price_chart, use_container_width=True, key="price_chart_main")
    
    # 기간 선택
    st.subheader("기간별 분석")
    periods = create_period_selector()
    
    if periods != "전체":
        days = get_period_days(periods)
        if days:
            recent_df = df.tail(days)
            recent_chart = create_price_chart(recent_df, price_col, chart_style)
            st.plotly_chart(recent_chart, use_container_width=True, key="price_chart_recent")


def render_technical_indicators_tab(df, price_col, show_technical_indicators):
    """기술지표 탭 렌더링"""
    st.header("기술지표")
    
    if show_technical_indicators:
        # 선택된 지표에 대한 기술지표 생성
        df_with_tech = df.copy()
        
        # 기술지표가 없는 경우 생성
        if 'rsi' not in df_with_tech.columns:
            delta = df_with_tech[price_col].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df_with_tech['rsi'] = 100 - (100 / (1 + rs))
        
        if 'bb_percent' not in df_with_tech.columns:
            ma20 = df_with_tech[price_col].rolling(20).mean()
            std20 = df_with_tech[price_col].rolling(20).std()
            df_with_tech['bb_percent'] = (df_with_tech[price_col] - (ma20 - 2 * std20)) / (4 * std20)
        
        if 'ma_ratio' not in df_with_tech.columns:
            ma20 = df_with_tech[price_col].rolling(20).mean()
            ma50 = df_with_tech[price_col].rolling(50).mean()
            df_with_tech['ma_ratio'] = ma20 / ma50
        
        tech_chart = create_technical_indicators_chart(df_with_tech)
        st.plotly_chart(tech_chart, use_container_width=True, key="technical_chart")
        
        # 기술지표 통계
        st.subheader(f"{price_col} 기술지표 통계")
        tech_stats_cols = ['rsi', 'bb_percent', 'ma_ratio']
        available_tech_cols = [col for col in tech_stats_cols if col in df_with_tech.columns]
        
        if available_tech_cols:
            tech_stats = df_with_tech[available_tech_cols].describe()
            st.dataframe(tech_stats)
        else:
            st.warning("기술지표 데이터가 없습니다.")
    else:
        st.info("사이드바에서 '기술지표 표시'를 체크하여 기술지표를 확인하세요.")
    
    display_technical_indicators_info()


def render_correlation_tab(df):
    """상관관계 탭 렌더링"""
    st.header("상관관계 분석")
    corr_chart = create_correlation_heatmap(df)
    if corr_chart:
        st.plotly_chart(corr_chart, use_container_width=True, key="correlation_chart")
    else:
        st.warning("상관관계 분석을 위한 충분한 데이터가 없습니다.")
    
    # 강한 상관관계 표시
    data_processor = DataProcessor()
    corr_matrix = data_processor.analyze_correlations(df)
    
    if corr_matrix is not None:
        strong_correlations = find_strong_correlations(corr_matrix)
        
        if strong_correlations:
            st.subheader("강한 상관관계 (|상관계수| >= 0.7)")
            for i, pair in enumerate(strong_correlations[:10], 1):
                direction = "양의" if pair['correlation'] > 0 else "음의"
                st.write(f"{i}. {direction} {pair['var1']} ↔ {pair['var2']}: {pair['correlation']:.3f}")


def render_statistics_tab(df, price_col, show_technical_indicators):
    """통계 탭 렌더링"""
    st.header("데이터 통계")
    
    # 탭 생성
    stat_tab1, stat_tab2, stat_tab3 = st.tabs(["📊 기본 통계", "📈 분포 분석", "📋 상세 정보"])
    
    with stat_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{price_col} 기본 통계")
            stats_df = df[price_col].describe()
            st.dataframe(stats_df)
            
            # 추가 통계 정보
            st.subheader("추가 통계")
            returns = df[price_col].pct_change().dropna()
            stats_measures = calculate_statistical_measures(returns, price_col)
            
            st.metric("연간 변동성", f"{stats_measures['volatility']:.2f}%")
            total_return = ((df[price_col].iloc[-1] / df[price_col].iloc[0]) - 1) * 100
            st.metric("총 수익률", f"{total_return:.2f}%")
        
        with col2:
            st.subheader("기술지표 통계")
            if show_technical_indicators:
                # 기술지표가 없는 경우 생성
                df_with_tech = df.copy()
                
                if 'rsi' not in df_with_tech.columns:
                    delta = df_with_tech[price_col].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    rs = gain / loss
                    df_with_tech['rsi'] = 100 - (100 / (1 + rs))
                
                if 'bb_percent' not in df_with_tech.columns:
                    ma20 = df_with_tech[price_col].rolling(20).mean()
                    std20 = df_with_tech[price_col].rolling(20).std()
                    df_with_tech['bb_percent'] = (df_with_tech[price_col] - (ma20 - 2 * std20)) / (4 * std20)
                
                if 'ma_ratio' not in df_with_tech.columns:
                    ma20 = df_with_tech[price_col].rolling(20).mean()
                    ma50 = df_with_tech[price_col].rolling(50).mean()
                    df_with_tech['ma_ratio'] = ma20 / ma50
                
                tech_stats_cols = ['rsi', 'bb_percent', 'ma_ratio']
                available_tech_cols = [col for col in tech_stats_cols if col in df_with_tech.columns]
                
                if available_tech_cols:
                    tech_stats = df_with_tech[available_tech_cols].describe()
                    st.dataframe(tech_stats)
                else:
                    st.warning("기술지표 데이터가 없습니다.")
            else:
                st.info("사이드바에서 '기술지표 표시'를 체크하여 기술지표 통계를 확인하세요.")
    
    with stat_tab2:
        st.subheader("분포 분석")
        
        # 수익률 계산
        returns = df[price_col].pct_change().dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 분포 분석 차트 생성
            fig_hist, fig_box, fig_qq = create_distribution_analysis(returns, price_col)
            st.plotly_chart(fig_hist, use_container_width=True, key="histogram")
            st.plotly_chart(fig_box, use_container_width=True, key="boxplot")
        
        with col2:
            st.plotly_chart(fig_qq, use_container_width=True, key="qqplot")
            
            # 정규성 검정 결과
            normality_results = get_normality_test_results(returns)
            
            st.subheader("정규성 검정")
            st.write(f"**Shapiro-Wilk 검정:**")
            st.write(f"  통계량: {normality_results['shapiro']['statistic']:.4f}")
            st.write(f"  p-value: {normality_results['shapiro']['p_value']:.4f}")
            st.write(f"  정규분포 여부: {'정규분포' if normality_results['shapiro']['is_normal'] else '비정규분포'}")
            
            st.write(f"**Jarque-Bera 검정:**")
            st.write(f"  통계량: {normality_results['jarque_bera']['statistic']:.4f}")
            st.write(f"  p-value: {normality_results['jarque_bera']['p_value']:.4f}")
            st.write(f"  정규분포 여부: {'정규분포' if normality_results['jarque_bera']['is_normal'] else '비정규분포'}")
    
    with stat_tab3:
        st.subheader("상세 정보")
        
        # 데이터 정보
        info_data = {
            "총 데이터 수": len(df),
            "분석 기간": f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}",
            "컬럼 수": len(df.columns),
            "결측값": df.isnull().sum().sum(),
            "분석 지표": price_col
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in info_data.items():
                st.metric(key, value)
        
        with col2:
            # 기술적 통계
            st.subheader("기술적 통계")
            
            # 왜도와 첨도
            returns = df[price_col].pct_change().dropna()
            stats_measures = calculate_statistical_measures(returns, price_col)
            
            st.metric("왜도 (Skewness)", f"{stats_measures['skewness']:.4f}")
            st.metric("첨도 (Kurtosis)", f"{stats_measures['kurtosis']:.4f}")
            st.metric("VaR (95%)", f"{stats_measures['var_95']:.4f}")
            st.metric("VaR (99%)", f"{stats_measures['var_99']:.4f}")
        
        # 모든 컬럼 정보
        st.subheader("전체 컬럼 정보")
        
        # 수치형 컬럼만 선택
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            # 상관관계 히트맵
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True, key="corr_heatmap")
        else:
            st.warning("수치형 컬럼이 없습니다.")


def render_ml_tab(df, price_col, enable_ml, ml_settings):
    """머신러닝 탭 렌더링"""
    st.header("머신러닝 모델")
    
    if enable_ml:
        # Optuna 최적화 실행
        if ml_settings.get('optimize_hyperparams'):
            with st.spinner("Optuna로 하이퍼파라미터를 최적화하고 있습니다..."):
                try:
                    # 데이터 준비
                    df_ml, feature_cols, y = prepare_ml_data(df, price_col)
                    X = df_ml[feature_cols]
                    
                    # 최적화 실행
                    best_params, best_score, optimization_history = optimize_hyperparameters(
                        X, y, ml_settings.get('n_trials', 50)
                    )
                    
                    # 최적화 결과 저장
                    st.session_state.best_params = best_params
                    st.session_state.optimization_history = optimization_history
                    
                    # 결과 표시
                    st.success("✅ Optuna 최적화 완료!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("최적 하이퍼파라미터")
                        for param, value in best_params.items():
                            st.metric(param, value)
                        
                        st.metric("최고 R² 점수", f"{best_score:.4f}")
                    
                    with col2:
                        st.subheader("최적화 통계")
                        st.write(f"총 시도 횟수: {len(optimization_history)}")
                    
                    # 최적화 과정 시각화
                    st.subheader("최적화 과정")
                    
                    # 목적 함수 값 변화
                    fig_optimization = create_optimization_process_chart(optimization_history)
                    st.plotly_chart(fig_optimization, use_container_width=True, key="optimization_process")
                    
                    # 하이퍼파라미터 중요도
                    if len(optimization_history) > 10:
                        import optuna
                        study = optuna.create_study()
                        study.add_trials(optimization_history)
                        importance = optuna.importance.get_param_importances(study)
                        
                        fig_importance_optuna = create_parameter_importance_chart(importance)
                        st.plotly_chart(fig_importance_optuna, use_container_width=True, key="param_importance")
                    
                except ImportError:
                    st.error("Optuna가 설치되지 않았습니다. 다음 명령어로 설치하세요:")
                    st.code("pip install optuna")
                except Exception as e:
                    st.error(f"최적화 중 오류가 발생했습니다: {e}")
        
        # 기존 최적화 결과 표시
        if st.session_state.get('best_params') is not None:
            st.subheader("이전 최적화 결과")
            st.write("**최적 하이퍼파라미터:**")
            for param, value in st.session_state.best_params.items():
                st.write(f"- {param}: {value}")
            
            if st.session_state.get('optimization_history') is not None:
                st.write("**최적화 히스토리:**")
                st.dataframe(st.session_state.optimization_history.head(10))
        
        if ml_settings.get('train_model'):
            # 학습 진행 상황 표시
            st.info("🤖 모델 학습을 시작합니다...")
            
            # 진행 상황 표시
            progress_bar, status_text = create_progress_display()
            
            try:
                # 데이터 준비
                update_progress(progress_bar, status_text, 10, "📊 데이터를 준비하고 있습니다...")
                df_ml, feature_cols, y = prepare_ml_data(df, price_col)
                X = df_ml[feature_cols]
                
                update_progress(progress_bar, status_text, 30, "🔀 데이터를 분할하고 있습니다...")
                
                # 데이터 분할
                test_size = ml_settings.get('test_size', 0.2)
                split_idx = int(len(X) * (1 - test_size))
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                update_progress(progress_bar, status_text, 50, "⚖️ 데이터를 스케일링하고 있습니다...")
                
                # 모델 학습
                update_progress(progress_bar, status_text, 70, "🌲 랜덤 포레스트 모델을 학습하고 있습니다...")
                
                model, metrics, y_pred = train_random_forest_model(
                    X_train, X_test, y_train, y_test,
                    ml_settings.get('n_estimators', 100),
                    ml_settings.get('max_depth', 10),
                    ml_settings.get('min_samples_split', 2),
                    ml_settings.get('min_samples_leaf', 1)
                )
                
                update_progress(progress_bar, status_text, 100, "✅ 모델 학습 완료!")
                clear_progress(progress_bar, status_text)
                
                # 결과를 session_state에 저장 (메인 페이지 표시용)
                st.session_state.model_results = {
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'r2': metrics['r2'],
                    'rmse': metrics['rmse'],
                    'mse': metrics['mse'],
                    'mae': metrics['mae']
                }
                
                # 결과 표시
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("모델 성능")
                    st.metric("MSE", f"{metrics['mse']:.6f}")
                    st.metric("MAE", f"{metrics['mae']:.6f}")
                    st.metric("R²", f"{metrics['r2']:.4f}")
                    st.metric("RMSE", f"{metrics['rmse']:.6f}")
                
                with col2:
                    st.subheader("하이퍼파라미터")
                    st.write(f"트리 개수: {ml_settings.get('n_estimators', 'N/A')}")
                    st.write(f"최대 깊이: {ml_settings.get('max_depth', 'N/A')}")
                    st.write(f"분할 최소 샘플 수: {ml_settings.get('min_samples_split', 'N/A')}")
                    st.write(f"리프 최소 샘플 수: {ml_settings.get('min_samples_leaf', 'N/A')}")
                    st.write(f"테스트 데이터 비율: {ml_settings.get('test_size', 'N/A'):.1%}")
                
                # 특성 중요도
                feature_importance = get_feature_importance(model, feature_cols)
                
                st.subheader("특성 중요도")
                fig_importance = create_feature_importance_chart(feature_importance, "상위 10개 특성 중요도")
                st.plotly_chart(fig_importance, use_container_width=True, key="feature_importance")
                
                # 예측 vs 실제
                fig_pred = create_prediction_scatter(y_test, y_pred)
                st.plotly_chart(fig_pred, use_container_width=True, key="pred_vs_actual")
                
                # 시계열 예측 결과
                fig_ts = create_time_series_prediction(y_test, y_pred)
                st.plotly_chart(fig_ts, use_container_width=True, key="time_series_pred")
                
                # 잔차 분석
                residuals = y_test.values - y_pred
                fig_residual, fig_residual_ts = create_residual_analysis(residuals)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(fig_residual, use_container_width=True, key="residual_hist")
                
                with col2:
                    st.plotly_chart(fig_residual_ts, use_container_width=True, key="residual_ts")
                
                # 추가 시각화
                st.subheader("추가 분석")
                
                # 학습 곡선 (가능한 경우)
                if hasattr(model, 'estimators_'):
                    # 각 트리의 성능 추정
                    train_scores, test_scores, x_values = calculate_learning_curve(
                        X_train, X_test, y_train, y_test,
                        ml_settings.get('max_depth', 10),
                        ml_settings.get('min_samples_split', 2),
                        ml_settings.get('min_samples_leaf', 1),
                        ml_settings.get('n_estimators', 100)
                    )
                    
                    fig_learning = create_learning_curve(train_scores, test_scores, x_values)
                    st.plotly_chart(fig_learning, use_container_width=True, key="learning_curve")
                
                # 예측 신뢰도 분석
                if hasattr(model, 'estimators_'):
                    # 앙상블 예측의 표준편차 계산
                    scaler = StandardScaler()
                    X_test_scaled = scaler.fit_transform(X_test)
                    mean_pred, std_pred = calculate_ensemble_predictions(model, X_test_scaled)
                    
                    fig_confidence = create_confidence_analysis(y_test, mean_pred, std_pred)
                    st.plotly_chart(fig_confidence, use_container_width=True, key="confidence_analysis")
                
                # 성능 요약 테이블
                st.subheader("모델 성능 요약")
                performance_summary = pd.DataFrame({
                    '지표': ['MSE', 'MAE', 'R²', 'RMSE', '설명된 분산 비율'],
                    '값': [f"{metrics['mse']:.6f}", f"{metrics['mae']:.6f}", f"{metrics['r2']:.4f}", f"{metrics['rmse']:.6f}", f"{metrics['r2']*100:.2f}%"]
                })
                st.table(performance_summary)
                
                # 모델 해석
                st.subheader("모델 해석")
                if metrics['r2'] > 0.7:
                    st.success("✅ 모델이 데이터를 잘 설명하고 있습니다 (R² > 0.7)")
                elif metrics['r2'] > 0.5:
                    st.warning("⚠️ 모델이 보통 수준의 성능을 보입니다 (0.5 < R² < 0.7)")
                else:
                    st.error("❌ 모델 성능이 낮습니다 (R² < 0.5)")
                
                # 개선 제안
                st.subheader("모델 개선 제안")
                
                if metrics['r2'] < 0.5:
                    st.write("• 더 많은 피처 추가 고려")
                    st.write("• 하이퍼파라미터 튜닝")
                    st.write("• 다른 모델 (XGBoost, LSTM 등) 시도")
                
                if len(feature_importance) > 0:
                    top_feature = feature_importance.iloc[0]['feature']
                    top_importance = feature_importance.iloc[0]['importance']
                    st.write(f"• 가장 중요한 피처: {top_feature} (중요도: {top_importance:.3f})")
                
            except Exception as e:
                st.error(f"모델 학습 중 오류가 발생했습니다: {e}")
                clear_progress(progress_bar, status_text)
    
    else:
        st.info("사이드바에서 '머신러닝 모델 활성화'를 체크하고 하이퍼파라미터를 설정한 후 '모델 학습 시작' 버튼을 클릭하세요.")
        display_ml_model_info()


def render_settings_tab(df, data_dir, price_col, enable_ml, ml_settings):
    """설정 탭 렌더링"""
    st.header("설정")
    
    # 데이터 새로고침
    if st.button("데이터 새로고침"):
        st.cache_data.clear()
        st.rerun()
    
    # 데이터 다운로드
    create_data_download_section(df)
    
    # 설정 정보
    display_system_info(df, data_dir, price_col, enable_ml)
    
    # 현재 설정 정보
    if enable_ml:
        display_ml_settings(ml_settings)


def main():
    st.title("📈 비트코인 데이터 분석 대시보드")
    st.markdown("---")
    
    # 데이터 로드 (사이드바보다 먼저)
    with st.spinner("데이터를 로드하고 있습니다..."):
        df = load_data()
    
    if df is None:
        st.error("데이터를 로드할 수 없습니다. 데이터 파일이 있는지 확인해주세요.")
        return
    
    # 기술지표 생성
    df = create_technical_indicators(df)
    
    # 날짜 컬럼 처리
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        df['date'] = pd.to_datetime(df.iloc[:, 0])
    
    # 사이드바 설정
    settings = create_sidebar_settings(df)
    selected_indicator = settings['selected_indicator']
    show_technical_indicators = settings['show_technical_indicators']
    chart_style = settings['chart_style']
    enable_ml = settings['enable_ml']
    ml_settings = settings['ml_settings']
    
    # 가격 컬럼 찾기 - 선택된 지표 사용
    price_col = selected_indicator if selected_indicator else None
    
    if price_col is None:
        st.error("가격 데이터를 찾을 수 없습니다.")
        return
    
    # 메인 대시보드
    create_metrics_display(df, price_col)
    
    st.markdown("---")
    
    # 모델 예측 결과가 있는 경우 메인 페이지에 표시
    if 'model_results' in st.session_state and st.session_state.model_results is not None:
        display_model_results(st.session_state.model_results)
        
        # 예측 vs 실제 차트
        col1, col2 = st.columns(2)
        
        with col1:
            # 산점도
            fig_pred_scatter = create_prediction_scatter(
                st.session_state.model_results['y_test'],
                st.session_state.model_results['y_pred']
            )
            st.plotly_chart(fig_pred_scatter, use_container_width=True, key="pred_scatter_main")
        
        with col2:
            # 시계열 예측
            fig_pred_ts = create_time_series_prediction(
                st.session_state.model_results['y_test'],
                st.session_state.model_results['y_pred']
            )
            st.plotly_chart(fig_pred_ts, use_container_width=True, key="pred_ts_main")
        
        # 예측 신뢰도 분석
        st.subheader("📊 예측 신뢰도 분석")
        
        # 잔차 분석
        residuals = st.session_state.model_results['y_test'].values - st.session_state.model_results['y_pred']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 잔차 분포
            fig_residual_main, _ = create_residual_analysis(residuals)
            st.plotly_chart(fig_residual_main, use_container_width=True, key="residual_main")
        
        with col2:
            # 잔차 시계열
            _, fig_residual_ts_main = create_residual_analysis(residuals)
            st.plotly_chart(fig_residual_ts_main, use_container_width=True, key="residual_ts_main")
        
        st.markdown("---")
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 가격 차트", 
        "📈 기술지표", 
        "🔥 상관관계", 
        "📋 데이터 통계",
        "🤖 머신러닝",
        "⚙️ 설정"
    ])
    
    with tab1:
        render_price_chart_tab(df, price_col, chart_style)
    
    with tab2:
        render_technical_indicators_tab(df, price_col, show_technical_indicators)
    
    with tab3:
        render_correlation_tab(df)
    
    with tab4:
        render_statistics_tab(df, price_col, show_technical_indicators)
    
    with tab5:
        render_ml_tab(df, price_col, enable_ml, ml_settings)
    
    with tab6:
        render_settings_tab(df, DATA_DIR, price_col, enable_ml, ml_settings)


if __name__ == "__main__":
    import os
    main() 