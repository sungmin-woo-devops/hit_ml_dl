"""
비트코인 데이터 시각화 모듈
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, jarque_bera


def create_price_chart(df: pd.DataFrame, price_col: str, chart_style: str = "라인 차트") -> go.Figure:
    """가격 차트 생성"""
    fig = go.Figure()
    
    if chart_style == "라인 차트":
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[price_col],
            mode='lines',
            name='가격',
            line=dict(color='blue', width=2)
        ))
        
        # 이동평균 추가
        ma_20 = df[price_col].rolling(20).mean()
        ma_50 = df[price_col].rolling(50).mean()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=ma_20,
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=ma_50,
            mode='lines',
            name='MA50',
            line=dict(color='red', width=1)
        ))
        
    elif chart_style == "캔들스틱 차트":
        # 캔들스틱 차트를 위한 데이터 준비
        df_candlestick = df.copy()
        df_candlestick['high'] = df_candlestick[price_col] * 1.02
        df_candlestick['low'] = df_candlestick[price_col] * 0.98
        df_candlestick['open'] = df_candlestick[price_col].shift(1)
        df_candlestick['close'] = df_candlestick[price_col]
        
        fig.add_trace(go.Candlestick(
            x=df_candlestick['date'],
            open=df_candlestick['open'],
            high=df_candlestick['high'],
            low=df_candlestick['low'],
            close=df_candlestick['close'],
            name='가격'
        ))
        
    elif chart_style == "영역 차트":
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[price_col],
            mode='lines',
            name='가격',
            fill='tonexty',
            line=dict(color='blue', width=2)
        ))
    
    fig.update_layout(
        title=f'{price_col} 가격 차트 ({chart_style})',
        xaxis_title='날짜',
        yaxis_title='가격',
        hovermode='x unified'
    )
    
    return fig


def create_technical_indicators_chart(df: pd.DataFrame) -> go.Figure:
    """기술지표 차트 생성"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('RSI', '볼린저 밴드 %B', '이동평균 비율'),
        vertical_spacing=0.1
    )
    
    # 정리된 데이터의 기술지표 컬럼 확인
    rsi_col = 'rsi_x' if 'rsi_x' in df.columns else 'rsi'
    bb_col = 'bb_percent_x' if 'bb_percent_x' in df.columns else 'bb_percent'
    ma_col = 'ma_ratio_x' if 'ma_ratio_x' in df.columns else 'ma_ratio'
    
    # RSI
    if rsi_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df[rsi_col], name='RSI', line=dict(color='purple')),
            row=1, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
    
    # 볼린저 밴드 %B
    if bb_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df[bb_col], name='BB %B', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="green", row=2, col=1)
    
    # 이동평균 비율
    if ma_col in df.columns:
        fig.add_trace(
            go.Scatter(x=df['date'], y=df[ma_col], name='MA Ratio', line=dict(color='blue')),
            row=3, col=1
        )
        fig.add_hline(y=1, line_dash="dash", line_color="black", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """상관관계 히트맵 생성"""
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="상관관계 히트맵"
    )
    return fig


def create_prediction_scatter(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """예측 vs 실제 산점도 생성"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test.values,
        y=y_pred,
        mode='markers',
        name='예측 vs 실제',
        marker=dict(color='blue', size=6, opacity=0.7)
    ))
    
    # 대각선 추가
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='완벽한 예측',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title="예측 vs 실제 (산점도)",
        xaxis_title="실제값",
        yaxis_title="예측값",
        showlegend=True
    )
    return fig


def create_time_series_prediction(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """시계열 예측 차트 생성"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=y_test.values,
        mode='lines',
        name='실제값',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(y_pred))),
        y=y_pred,
        mode='lines',
        name='예측값',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title="시계열 예측 결과",
        xaxis_title="시간",
        yaxis_title="가격",
        showlegend=True
    )
    return fig


def create_residual_analysis(residuals: np.ndarray) -> tuple[go.Figure, go.Figure]:
    """잔차 분석 차트 생성"""
    # 잔차 분포
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=residuals,
        nbinsx=20,
        name='잔차 분포',
        marker_color='lightgreen',
        opacity=0.7
    ))
    fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
    fig_hist.update_layout(
        title="잔차 분포",
        xaxis_title="잔차",
        yaxis_title="빈도"
    )
    
    # 잔차 시계열
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=list(range(len(residuals))),
        y=residuals,
        mode='lines',
        name='잔차',
        line=dict(color='orange', width=1)
    ))
    fig_ts.add_hline(y=0, line_dash="dash", line_color="red")
    fig_ts.update_layout(
        title="잔차 시계열",
        xaxis_title="시간",
        yaxis_title="잔차"
    )
    
    return fig_hist, fig_ts


def create_distribution_analysis(returns: pd.Series, price_col: str) -> tuple[go.Figure, go.Figure, go.Figure]:
    """분포 분석 차트 생성"""
    # 히스토그램
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=returns,
        nbinsx=30,
        name='수익률 분포',
        marker_color='lightblue'
    ))
    fig_hist.update_layout(
        title=f"{price_col} 수익률 분포",
        xaxis_title="수익률",
        yaxis_title="빈도",
        showlegend=False
    )
    
    # 박스플롯
    fig_box = go.Figure()
    fig_box.add_trace(go.Box(
        y=returns,
        name='수익률',
        marker_color='lightgreen'
    ))
    fig_box.update_layout(
        title=f"{price_col} 수익률 박스플롯",
        yaxis_title="수익률",
        showlegend=False
    )
    
    # Q-Q 플롯
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
    sample_quantiles = np.sort(returns)
    
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        name='Q-Q Plot',
        marker=dict(color='red', size=6)
    ))
    
    # 대각선 추가
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig_qq.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='정규분포',
        line=dict(color='black', dash='dash')
    ))
    
    fig_qq.update_layout(
        title=f"{price_col} Q-Q Plot (정규분포 비교)",
        xaxis_title="이론적 분위수",
        yaxis_title="표본 분위수",
        showlegend=True
    )
    
    return fig_hist, fig_box, fig_qq


def create_feature_importance_chart(feature_importance: pd.DataFrame, title: str = "특성 중요도") -> go.Figure:
    """특성 중요도 차트 생성"""
    fig = px.bar(
        feature_importance.head(10),
        x='importance',
        y='feature',
        orientation='h',
        title=title
    )
    return fig


def create_learning_curve(train_scores: list, test_scores: list, x_values: list) -> go.Figure:
    """학습 곡선 차트 생성"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=train_scores,
        mode='lines',
        name='훈련 R²',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=x_values,
        y=test_scores,
        mode='lines',
        name='테스트 R²',
        line=dict(color='red')
    ))
    fig.update_layout(
        title="학습 곡선",
        xaxis_title="트리 개수",
        yaxis_title="R² 점수"
    )
    return fig


def create_confidence_analysis(y_test: pd.Series, mean_pred: np.ndarray, std_pred: np.ndarray) -> go.Figure:
    """예측 신뢰도 분석 차트 생성"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(y_test))),
        y=y_test.values,
        mode='lines',
        name='실제',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(mean_pred))),
        y=mean_pred,
        mode='lines',
        name='예측',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(mean_pred))),
        y=mean_pred + 2*std_pred,
        mode='lines',
        name='신뢰구간 상단',
        line=dict(color='gray', dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(mean_pred))),
        y=mean_pred - 2*std_pred,
        mode='lines',
        name='신뢰구간 하단',
        line=dict(color='gray', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.2)'
    ))
    fig.update_layout(
        title="예측 신뢰도 분석 (95% 신뢰구간)",
        xaxis_title="시간",
        yaxis_title="가격"
    )
    return fig


def create_optimization_process_chart(trials: list) -> go.Figure:
    """최적화 과정 차트 생성"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(trials))),
        y=[trial.value for trial in trials if trial.value is not None],
        mode='lines+markers',
        name='R² 점수',
        line=dict(color='blue'),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title="Optuna 최적화 과정",
        xaxis_title="시도 횟수",
        yaxis_title="R² 점수"
    )
    return fig


def create_parameter_importance_chart(importance: dict) -> go.Figure:
    """하이퍼파라미터 중요도 차트 생성"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(importance.values()),
        y=list(importance.keys()),
        orientation='h',
        marker_color='lightcoral'
    ))
    fig.update_layout(
        title="하이퍼파라미터 중요도",
        xaxis_title="중요도",
        yaxis_title="하이퍼파라미터"
    )
    return fig 