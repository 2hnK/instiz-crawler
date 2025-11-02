"""
기상-감성 연구를 위한 감성 분석 가이드
"""

# ============================================================================
# 방법 1: KoBERT 기반 감성 분류 (추천 ⭐⭐⭐)
# ============================================================================

"""
장점:
- 높은 정확도 (85-90%)
- 한국어 구어체/줄임말 잘 처리
- 사전 학습된 모델 활용 가능
- 문맥 이해 능력

단점:
- 연산 비용 높음 (GPU 권장)
- 초기 세팅 복잡

추천 모델:
1. KcELECTRA (가장 가벼우면서 성능 좋음)
2. KoBERT
3. KoGPT
"""

# 설치
# pip install transformers torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def kobert_sentiment_analysis(text):
    """
    KoBERT 기반 3-class 감성 분류
    Returns: {'label': 'positive'/'negative'/'neutral', 'score': 0.0-1.0}
    """
    # 사전 학습된 모델 로드
    model_name = "beomi/KcELECTRA-base-v2022"  # 또는 다른 KoBERT 모델
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # 입력 토큰화
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_class = torch.argmax(predictions).item()
    confidence = predictions[0][predicted_class].item()
    
    return {
        'label': label_map[predicted_class],
        'score': confidence,
        'all_scores': {
            'negative': predictions[0][0].item(),
            'neutral': predictions[0][1].item(),
            'positive': predictions[0][2].item()
        }
    }


# ============================================================================
# 방법 2: KNU 감성 사전 기반 (추천 ⭐⭐)
# ============================================================================

"""
장점:
- 빠른 처리 속도
- 구현 단순
- 해석 용이
- GPU 불필요

단점:
- 문맥 미고려
- 신조어/줄임말 처리 약함
- 정확도 상대적으로 낮음 (70-80%)

적합한 경우:
- 대량 데이터 빠르게 처리
- 보조 지표로 사용
"""

# 설치
# pip install soynlp

from soynlp.normalizer import repeat_normalize
import re

class KNUSentimentAnalyzer:
    """KNU 감성 사전 기반 분석기"""
    
    def __init__(self):
        # KNU 감성 사전 로드 (파일 필요)
        # http://dilab.kunsan.ac.kr/knusl.html 에서 다운로드
        self.positive_words = self._load_lexicon('positive.txt')
        self.negative_words = self._load_lexicon('negative.txt')
    
    def _load_lexicon(self, filepath):
        """감성 사전 로드"""
        words = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0]
                    score = float(parts[1])
                    words[word] = score
        return words
    
    def preprocess(self, text):
        """텍스트 전처리"""
        # 반복 문자 정규화 (ㅋㅋㅋㅋ -> ㅋㅋ)
        text = repeat_normalize(text, num_repeats=2)
        # 특수문자 제거 (선택적)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def analyze(self, text):
        """
        감성 점수 계산
        Returns: {'score': -1.0 ~ 1.0, 'label': str, 'details': dict}
        """
        text = self.preprocess(text)
        words = text.split()
        
        pos_score = 0
        neg_score = 0
        pos_count = 0
        neg_count = 0
        
        for word in words:
            if word in self.positive_words:
                pos_score += self.positive_words[word]
                pos_count += 1
            if word in self.negative_words:
                neg_score += self.negative_words[word]
                neg_count += 1
        
        # 점수 정규화
        total_count = max(pos_count + neg_count, 1)
        normalized_score = (pos_score - neg_score) / total_count
        
        # 라벨 할당
        if normalized_score > 0.1:
            label = 'positive'
        elif normalized_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': normalized_score,
            'label': label,
            'details': {
                'positive_words': pos_count,
                'negative_words': neg_count,
                'pos_score': pos_score,
                'neg_score': neg_score
            }
        }


# ============================================================================
# 방법 3: 앙상블 접근 (최고 정확도 ⭐⭐⭐⭐)
# ============================================================================

"""
KoBERT + 감성 사전 조합
- 두 방법의 장점 결합
- 신뢰도 높은 결과
"""

class EnsembleSentimentAnalyzer:
    """앙상블 감성 분석기"""
    
    def __init__(self):
        self.kobert = kobert_sentiment_analysis  # 위의 함수
        self.knu = KNUSentimentAnalyzer()
    
    def analyze(self, text):
        """
        두 모델의 결과를 종합
        Returns: 최종 감성 점수 및 라벨
        """
        # KoBERT 결과
        bert_result = self.kobert(text)
        bert_score = (
            bert_result['all_scores']['positive'] - 
            bert_result['all_scores']['negative']
        )
        
        # KNU 결과
        knu_result = self.knu.analyze(text)
        knu_score = knu_result['score']
        
        # 가중 평균 (KoBERT에 더 높은 가중치)
        final_score = 0.7 * bert_score + 0.3 * knu_score
        
        # 최종 라벨
        if final_score > 0.1:
            label = 'positive'
        elif final_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': final_score,
            'label': label,
            'bert_result': bert_result,
            'knu_result': knu_result,
            'confidence': (bert_result['score'] + abs(knu_score)) / 2
        }


# ============================================================================
# 실제 데이터셋 적용 예시
# ============================================================================

import pandas as pd
from tqdm import tqdm

def process_instiz_data(df):
    """
    Instiz 데이터프레임에 감성 분석 적용
    
    Args:
        df: columns=['id', 'title', 'body', 'created_at', ...]
    
    Returns:
        df with added columns: sentiment_label, sentiment_score
    """
    analyzer = EnsembleSentimentAnalyzer()  # 또는 다른 분석기
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 제목 + 본문 결합
        text = str(row['title']) + ' ' + str(row['body'])
        
        # 감성 분석
        try:
            result = analyzer.analyze(text)
            results.append({
                'sentiment_label': result['label'],
                'sentiment_score': result['score']
            })
        except Exception as e:
            # 오류 시 중립으로 처리
            results.append({
                'sentiment_label': 'neutral',
                'sentiment_score': 0.0
            })
    
    # 결과 병합
    result_df = pd.DataFrame(results)
    df = pd.concat([df, result_df], axis=1)
    
    return df


# ============================================================================
# 감성 지수 집계 (일별/시간별)
# ============================================================================

def calculate_sentiment_index(df, groupby='date'):
    """
    감성 지수 계산
    
    Args:
        df: 감성 분석 완료된 데이터프레임
        groupby: 'date', 'hour', 'date_hour' 등
    
    Returns:
        집계된 감성 지수
    """
    if groupby == 'date':
        df['date'] = pd.to_datetime(df['created_at']).dt.date
        group_col = 'date'
    elif groupby == 'hour':
        df['hour'] = pd.to_datetime(df['created_at']).dt.hour
        group_col = 'hour'
    elif groupby == 'date_hour':
        df['date_hour'] = pd.to_datetime(df['created_at']).dt.floor('H')
        group_col = 'date_hour'
    
    # 집계
    sentiment_index = df.groupby(group_col).agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'sentiment_label': lambda x: (x == 'positive').sum() / len(x)  # 긍정 비율
    }).reset_index()
    
    sentiment_index.columns = [
        group_col, 
        'avg_sentiment', 
        'sentiment_std', 
        'post_count',
        'positive_ratio'
    ]
    
    return sentiment_index


# ============================================================================
# 성능 평가 (선택)
# ============================================================================

def validate_sentiment_model(sample_size=100):
    """
    랜덤 샘플을 수동 라벨링하여 정확도 측정
    
    연구의 신뢰도를 높이기 위해 권장
    """
    # 1. 랜덤 샘플 추출
    # 2. 수동으로 긍정/부정/중립 라벨링
    # 3. 모델 예측과 비교
    # 4. 정확도, F1-score 계산
    
    pass  # 구체적 구현은 실제 데이터로


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 데이터 로드
    posts = pd.read_csv('instiz_posts.csv')
    
    # 감성 분석 수행
    posts_with_sentiment = process_instiz_data(posts)
    
    # 일별 감성 지수 계산
    daily_sentiment = calculate_sentiment_index(posts_with_sentiment, groupby='date')
    
    # 저장
    daily_sentiment.to_csv('daily_sentiment_index.csv', index=False)
    
    print("감성 분석 완료!")
    print(f"평균 감성 점수: {posts_with_sentiment['sentiment_score'].mean():.3f}")
    print(f"긍정 비율: {(posts_with_sentiment['sentiment_label']=='positive').mean()*100:.1f}%")
    print(f"부정 비율: {(posts_with_sentiment['sentiment_label']=='negative').mean()*100:.1f}%")
