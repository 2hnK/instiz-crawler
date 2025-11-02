"""
기상요인-감성지수 연구를 위한 통계 분석 가이드
"""

# ============================================================================
# 분석 1: 기술 통계 및 탐색적 데이터 분석 (EDA)
# ============================================================================

"""
목적: 데이터의 기본적인 특성 파악
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def eda_analysis(df_sentiment, df_weather):
    """
    탐색적 데이터 분석
    
    Args:
        df_sentiment: 감성 분석 완료된 게시물 데이터
        df_weather: 기상 데이터
    """
    
    # 1. 기본 통계량
    print("=" * 70)
    print("1. 감성 지수 기술통계")
    print("=" * 70)
    print(df_sentiment['sentiment_score'].describe())
    
    # 2. 시계열 플롯
    plt.figure(figsize=(15, 6))
    
    plt.subplot(2, 1, 1)
    df_sentiment.groupby('date')['sentiment_score'].mean().plot()
    plt.title('Daily Average Sentiment Score')
    plt.ylabel('Sentiment Score')
    
    plt.subplot(2, 1, 2)
    df_weather.groupby('date')['temperature'].mean().plot()
    plt.title('Daily Average Temperature')
    plt.ylabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig('timeseries_plot.png', dpi=300)
    
    # 3. 분포 확인
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(df_sentiment['sentiment_score'], bins=50, edgecolor='black')
    plt.title('Sentiment Score Distribution')
    plt.xlabel('Sentiment Score')
    
    plt.subplot(1, 3, 2)
    plt.hist(df_weather['temperature'], bins=30, edgecolor='black')
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    
    plt.subplot(1, 3, 3)
    plt.hist(df_weather['precipitation'], bins=30, edgecolor='black')
    plt.title('Precipitation Distribution')
    plt.xlabel('Precipitation (mm)')
    
    plt.tight_layout()
    plt.savefig('distribution_plots.png', dpi=300)
    
    # 4. 계절성 확인
    df_sentiment['month'] = pd.to_datetime(df_sentiment['date']).dt.month
    df_sentiment.groupby('month')['sentiment_score'].mean().plot(kind='bar')
    plt.title('Average Sentiment by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment Score')
    plt.savefig('seasonal_pattern.png', dpi=300)


# ============================================================================
# 분석 2: 상관관계 분석 (Correlation Analysis)
# ============================================================================

"""
목적: 기상요인과 감성지수 간의 선형 관계 파악
방법: Pearson/Spearman 상관계수
"""

def correlation_analysis(merged_df):
    """
    상관관계 분석
    
    Args:
        merged_df: 감성 + 기상 데이터 병합된 데이터프레임
    """
    
    # 분석 변수 선택
    variables = [
        'sentiment_score',
        'temperature',
        'temp_change',  # 전일 대비 기온 변화
        'precipitation',
        'humidity',
        'sunshine_duration',
        'wind_speed',
        'temp_range'  # 일교차
    ]
    
    # Pearson 상관계수
    corr_pearson = merged_df[variables].corr(method='pearson')
    
    # Spearman 상관계수 (비선형 관계 고려)
    corr_spearman = merged_df[variables].corr(method='spearman')
    
    # 히트맵
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.heatmap(corr_pearson, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0], vmin=-1, vmax=1)
    axes[0].set_title('Pearson Correlation')
    
    sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', center=0,
                ax=axes[1], vmin=-1, vmax=1)
    axes[1].set_title('Spearman Correlation')
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    
    # 유의성 검정
    print("\n" + "=" * 70)
    print("상관관계 유의성 검정")
    print("=" * 70)
    
    for var in variables[1:]:  # sentiment_score 제외
        r, p = stats.pearsonr(merged_df['sentiment_score'], merged_df[var])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{var:20s}: r = {r:6.3f}, p = {p:.4f} {sig}")
    
    return corr_pearson, corr_spearman


# ============================================================================
# 분석 3: 다중 회귀 분석 (Multiple Regression) ⭐⭐⭐
# ============================================================================

"""
목적: 여러 기상요인이 감성에 미치는 영향을 동시에 분석
방법: OLS (Ordinary Least Squares) 회귀

장점:
- 여러 요인의 독립적 효과 파악
- 통제변수 포함 가능
- 해석이 직관적
"""

from statsmodels.api import OLS, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

def multiple_regression(merged_df):
    """
    다중 회귀 분석
    
    Model: sentiment = β0 + β1*temp + β2*precip + ... + ε
    """
    
    # 독립변수 (기상요인)
    X_vars = [
        'temperature',
        'temp_change',
        'precipitation',
        'humidity',
        'sunshine_duration',
        'temp_range'
    ]
    
    # 통제변수 추가
    control_vars = [
        'is_weekend',  # 주말 여부
        'is_holiday',  # 공휴일 여부
        'hour',  # 시간대
        'month'  # 월 (계절성)
    ]
    
    # 데이터 준비
    X = merged_df[X_vars + control_vars].copy()
    y = merged_df['sentiment_score'].copy()
    
    # 결측치 제거
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # 상수항 추가
    X = add_constant(X)
    
    # 회귀 모델 적합
    model = OLS(y, X).fit()
    
    # 결과 출력
    print("\n" + "=" * 70)
    print("다중 회귀 분석 결과")
    print("=" * 70)
    print(model.summary())
    
    # VIF (다중공선성 검사)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns[1:]  # 상수항 제외
    vif_data["VIF"] = [variance_inflation_factor(X.values, i+1) 
                       for i in range(len(X.columns)-1)]
    
    print("\n다중공선성 검사 (VIF):")
    print(vif_data)
    print("(VIF > 10이면 다중공선성 문제)")
    
    return model


# ============================================================================
# 분석 4: 패널 회귀 분석 (Panel Regression) ⭐⭐⭐⭐
# ============================================================================

"""
목적: 시간대별/날짜별 반복 측정 데이터의 구조 고려
방법: Fixed Effects / Random Effects Model

장점:
- 개체별(시간대별) 고정 효과 통제
- 더 정확한 인과 추론
- 시계열 특성 반영

추천도: ⭐⭐⭐⭐ (귀하의 연구에 가장 적합할 수 있음)
"""

from linearmodels import PanelOLS, RandomEffects

def panel_regression(merged_df):
    """
    패널 회귀 분석
    
    패널 구조: 날짜 x 시간대 (또는 날짜만)
    """
    
    # 패널 데이터 준비
    # entity: date, time: hour (또는 date만 사용)
    df_panel = merged_df.set_index(['date', 'hour'])
    
    # 독립변수
    exog_vars = [
        'temperature',
        'temp_change',
        'precipitation',
        'humidity',
        'sunshine_duration'
    ]
    
    # 통제변수
    control_vars = [
        'is_weekend',
        'is_holiday'
    ]
    
    # Fixed Effects Model
    print("\n" + "=" * 70)
    print("Fixed Effects Model")
    print("=" * 70)
    
    formula = 'sentiment_score ~ ' + ' + '.join(exog_vars + control_vars) + ' + EntityEffects'
    model_fe = PanelOLS.from_formula(formula, data=df_panel).fit(cov_type='clustered', cluster_entity=True)
    print(model_fe)
    
    # Random Effects Model
    print("\n" + "=" * 70)
    print("Random Effects Model")
    print("=" * 70)
    
    model_re = RandomEffects.from_formula(formula.replace('EntityEffects', ''), data=df_panel).fit()
    print(model_re)
    
    # Hausman Test (어떤 모델이 적합한지)
    # 생략 (필요시 구현)
    
    return model_fe, model_re


# ============================================================================
# 분석 5: 시계열 분석 (Time Series Analysis) ⭐⭐⭐
# ============================================================================

"""
목적: 시간적 의존성과 인과관계 탐색
방법: VAR (Vector Autoregression), Granger Causality

장점:
- 시간적 선후관계 분석
- 쌍방향 인과관계 검정
- 동적 효과 파악
"""

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

def time_series_analysis(df_daily):
    """
    시계열 분석
    
    Args:
        df_daily: 일별로 집계된 데이터
    """
    
    # VAR 모델을 위한 데이터 준비
    ts_data = df_daily[['sentiment_score', 'temperature', 'precipitation']].dropna()
    
    # VAR 모델
    model_var = VAR(ts_data)
    results_var = model_var.fit(maxlags=7)  # 최대 7일 지연
    
    print("\n" + "=" * 70)
    print("VAR Model Summary")
    print("=" * 70)
    print(results_var.summary())
    
    # Granger Causality Test
    print("\n" + "=" * 70)
    print("Granger Causality Test: 기온 → 감성")
    print("=" * 70)
    
    gc_test = grangercausalitytests(
        ts_data[['sentiment_score', 'temperature']], 
        maxlag=7,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Granger Causality Test: 강수량 → 감성")
    print("=" * 70)
    
    gc_test = grangercausalitytests(
        ts_data[['sentiment_score', 'precipitation']], 
        maxlag=7,
        verbose=True
    )
    
    return results_var


# ============================================================================
# 분석 6: 비선형 효과 분석 (선택)
# ============================================================================

"""
목적: 날씨의 극단값이 감성에 미치는 비선형 효과 분석
예: 너무 덥거나 추운 경우 부정적 감성

방법: Polynomial Regression, Spline Regression
"""

from sklearn.preprocessing import PolynomialFeatures
from scipy.interpolate import UnivariateSpline

def nonlinear_analysis(merged_df):
    """
    비선형 효과 분석
    """
    
    # 2차 다항식 회귀
    X = merged_df[['temperature']].values
    y = merged_df['sentiment_score'].values
    
    # 결측치 제거
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    # 다항 특성 생성
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # 회귀
    model_poly = OLS(y, X_poly).fit()
    
    print("\n" + "=" * 70)
    print("Polynomial Regression (Temperature^2)")
    print("=" * 70)
    print(model_poly.summary())
    
    # 시각화
    X_sorted = np.sort(X, axis=0)
    X_poly_sorted = poly.transform(X_sorted)
    y_pred = model_poly.predict(X_poly_sorted)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.3, label='Actual')
    plt.plot(X_sorted, y_pred, 'r-', linewidth=2, label='Fitted')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Sentiment Score')
    plt.title('Nonlinear Relationship: Temperature vs Sentiment')
    plt.legend()
    plt.savefig('nonlinear_temperature.png', dpi=300)
    
    return model_poly


# ============================================================================
# 분석 7: 조절 효과 분석 (Moderation Analysis)
# ============================================================================

"""
목적: 날씨 효과가 상황에 따라 달라지는지 분석
예: 주말과 평일에서 날씨 효과가 다른가?

방법: 상호작용항 포함 회귀
"""

def moderation_analysis(merged_df):
    """
    조절 효과 분석
    
    Model: sentiment = β0 + β1*temp + β2*weekend + β3*temp×weekend + ε
    """
    
    # 상호작용항 생성
    merged_df['temp_x_weekend'] = (
        merged_df['temperature'] * merged_df['is_weekend']
    )
    
    # 회귀 모델
    X_vars = [
        'temperature',
        'is_weekend',
        'temp_x_weekend',
        'precipitation',
        'hour'
    ]
    
    X = add_constant(merged_df[X_vars].dropna())
    y = merged_df.loc[X.index, 'sentiment_score']
    
    model_mod = OLS(y, X).fit()
    
    print("\n" + "=" * 70)
    print("Moderation Analysis: Temperature × Weekend")
    print("=" * 70)
    print(model_mod.summary())
    
    # 해석
    if model_mod.pvalues['temp_x_weekend'] < 0.05:
        print("\n✓ 유의미한 조절효과 발견!")
        print("  → 날씨가 감성에 미치는 영향이 주말/평일에 따라 다름")
    else:
        print("\n✗ 유의미한 조절효과 없음")
    
    return model_mod


# ============================================================================
# 종합 분석 파이프라인
# ============================================================================

def full_analysis_pipeline(df_posts, df_weather):
    """
    전체 분석 파이프라인
    """
    
    print("\n" + "=" * 70)
    print("기상요인-감성지수 통계 분석")
    print("=" * 70)
    
    # 1. 데이터 병합
    merged_df = merge_sentiment_weather(df_posts, df_weather)
    
    # 2. EDA
    print("\n[Step 1] 탐색적 데이터 분석")
    eda_analysis(df_posts, df_weather)
    
    # 3. 상관관계 분석
    print("\n[Step 2] 상관관계 분석")
    corr_p, corr_s = correlation_analysis(merged_df)
    
    # 4. 다중 회귀 (기본)
    print("\n[Step 3] 다중 회귀 분석")
    model_reg = multiple_regression(merged_df)
    
    # 5. 패널 회귀 (추천)
    print("\n[Step 4] 패널 회귀 분석")
    model_fe, model_re = panel_regression(merged_df)
    
    # 6. 시계열 분석
    print("\n[Step 5] 시계열 분석")
    df_daily = merged_df.groupby('date').mean()
    model_var = time_series_analysis(df_daily)
    
    # 7. 비선형 효과
    print("\n[Step 6] 비선형 효과 분석")
    model_poly = nonlinear_analysis(merged_df)
    
    # 8. 조절 효과
    print("\n[Step 7] 조절 효과 분석")
    model_mod = moderation_analysis(merged_df)
    
    print("\n" + "=" * 70)
    print("분석 완료!")
    print("=" * 70)
    
    return {
        'correlation': (corr_p, corr_s),
        'regression': model_reg,
        'panel_fe': model_fe,
        'panel_re': model_re,
        'var': model_var,
        'polynomial': model_poly,
        'moderation': model_mod
    }


def merge_sentiment_weather(df_posts, df_weather):
    """
    감성 데이터와 기상 데이터 병합
    """
    # 시간대별 병합 (hour 단위)
    df_posts['datetime'] = pd.to_datetime(df_posts['created_at'])
    df_posts['date'] = df_posts['datetime'].dt.date
    df_posts['hour'] = df_posts['datetime'].dt.hour
    
    df_weather['datetime'] = pd.to_datetime(df_weather['datetime'])
    df_weather['date'] = df_weather['datetime'].dt.date
    df_weather['hour'] = df_weather['datetime'].dt.hour
    
    # 병합
    merged = pd.merge(
        df_posts,
        df_weather,
        on=['date', 'hour'],
        how='left'
    )
    
    return merged
