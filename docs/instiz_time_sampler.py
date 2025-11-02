"""
Instiz 시간대별 균등 샘플링 크롤러
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
from urllib.parse import quote

class InstizTimeSampler:
    """시간대별 균등 샘플링 크롤러"""
    
    def __init__(self, target_per_timerange=1250):
        self.base_url = "https://www.instiz.net/name"
        self.target_per_timerange = target_per_timerange
        
        # 4개 시간대 (6시간씩)
        self.time_ranges = [
            ('00:00', '05:59', '새벽'),
            ('06:00', '11:59', '오전'),
            ('12:00', '17:59', '오후'),
            ('18:00', '23:59', '저녁')
        ]
    
    def build_url(self, year, month, start_time, end_time, page=1):
        """
        URL 생성
        
        Args:
            year: 2024
            month: 1
            start_time: '00:00'
            end_time: '05:59'
            page: 페이지 번호
        """
        # 날짜 범위
        start_date = f"{year}/{month:02d}/01 {start_time}"
        
        # 해당 월의 마지막 날 계산
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = (next_month - timedelta(days=1)).day
        end_date = f"{year}/{month:02d}/{last_day} {end_time}"
        
        # URL 인코딩
        start_encoded = quote(start_date)
        end_encoded = quote(end_date)
        
        url = (
            f"{self.base_url}?"
            f"page={page}&"
            f"category=1&"  # 일상 게시판
            f"k=%EA%B8%B0%EA%B0%84%ED%83%90%EC%83%89&"  # 기간탐색
            f"stype=9&"
            f"starttime={start_encoded}&"
            f"endtime={end_encoded}"
        )
        
        return url
    
    def get_total_posts_in_range(self, year, month, start_time, end_time):
        """
        특정 시간대의 전체 게시물 수 확인
        
        Returns:
            total_posts: 전체 게시물 수
            total_pages: 전체 페이지 수
        """
        url = self.build_url(year, month, start_time, end_time, page=1)
        
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # 페이지네이션에서 전체 게시물 수 추출
            # (실제 HTML 구조에 맞게 수정 필요)
            pagination = soup.find('div', class_='pagination')
            if pagination:
                # 예시: "1 / 2500" 형태
                total_posts = int(pagination.text.split('/')[-1].strip())
                total_pages = (total_posts // 20) + (1 if total_posts % 20 else 0)
                return total_posts, total_pages
            
        except Exception as e:
            print(f"Error: {e}")
            return None, None
    
    def calculate_sampling_pages(self, total_pages, target_count):
        """
        균등 간격으로 샘플링할 페이지 번호 계산
        
        Args:
            total_pages: 전체 페이지 수
            target_count: 목표 게시물 수
        
        Returns:
            list of page numbers to crawl
        """
        posts_per_page = 20
        target_pages = target_count // posts_per_page
        
        if target_pages >= total_pages:
            # 목표가 전체보다 많으면 모든 페이지 수집
            return list(range(1, total_pages + 1))
        
        # 균등 간격 계산
        interval = total_pages / target_pages
        pages = [int(i * interval) + 1 for i in range(target_pages)]
        
        return pages
    
    def crawl_page(self, url):
        """
        단일 페이지 크롤링
        
        Returns:
            list of dict: 게시물 정보
        """
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            posts = []
            # 실제 HTML 구조에 맞게 수정 필요
            post_elements = soup.find_all('div', class_='post-item')
            
            for post in post_elements:
                post_data = {
                    'id': post.get('data-id'),
                    'url': post.find('a')['href'],
                    'title': post.find('h3').text.strip(),
                    'body': post.find('div', class_='body').text.strip(),
                    'created_at': post.find('time')['datetime'],
                    'likes': int(post.find('span', class_='likes').text),
                    'comments_count': int(post.find('span', class_='comments').text)
                }
                posts.append(post_data)
            
            return posts
            
        except Exception as e:
            print(f"Error crawling page: {e}")
            return []
    
    def sample_month(self, year, month):
        """
        한 달 전체를 시간대별 균등 샘플링
        
        Args:
            year: 2024
            month: 1
        
        Returns:
            DataFrame with sampled posts
        """
        all_posts = []
        
        print(f"\n{'='*60}")
        print(f"수집 시작: {year}년 {month}월")
        print(f"{'='*60}")
        
        for start_time, end_time, label in self.time_ranges:
            print(f"\n📅 {label} ({start_time}-{end_time}) 수집 중...")
            
            # 1. 전체 게시물 수 확인
            total_posts, total_pages = self.get_total_posts_in_range(
                year, month, start_time, end_time
            )
            
            if total_posts is None:
                print(f"   ⚠️ 게시물 수 확인 실패")
                continue
            
            print(f"   전체 게시물: {total_posts:,}개 ({total_pages:,} 페이지)")
            
            # 2. 샘플링할 페이지 계산
            pages_to_crawl = self.calculate_sampling_pages(
                total_pages, 
                self.target_per_timerange
            )
            
            print(f"   수집 목표: {self.target_per_timerange:,}개")
            print(f"   크롤링 페이지: {len(pages_to_crawl):,}개")
            
            # 3. 페이지별 크롤링
            for i, page_num in enumerate(pages_to_crawl):
                if i > 0 and i % 10 == 0:
                    print(f"   진행: {i}/{len(pages_to_crawl)} 페이지...")
                
                url = self.build_url(year, month, start_time, end_time, page_num)
                posts = self.crawl_page(url)
                all_posts.extend(posts)
                
                # Rate limiting
                time.sleep(1)  # 1초 대기
            
            print(f"   ✅ {label} 완료: {len([p for p in all_posts if p.get('timerange')==label]):,}개 수집")
        
        # DataFrame 변환
        df = pd.DataFrame(all_posts)
        
        print(f"\n{'='*60}")
        print(f"수집 완료!")
        print(f"총 게시물: {len(df):,}개")
        print(f"시간대별 분포:")
        for _, _, label in self.time_ranges:
            count = len([p for p in all_posts if p.get('timerange') == label])
            print(f"   {label}: {count:,}개")
        print(f"{'='*60}\n")
        
        return df
    
    def sample_year(self, year, months=range(1, 13)):
        """
        1년 전체 수집
        
        Args:
            year: 2024
            months: 수집할 월 리스트 (기본값: 1-12월)
        
        Returns:
            DataFrame
        """
        all_data = []
        
        for month in months:
            print(f"\n{'#'*60}")
            print(f"# {year}년 {month}월 수집")
            print(f"{'#'*60}")
            
            monthly_data = self.sample_month(year, month)
            all_data.append(monthly_data)
            
            # 월별 저장 (백업)
            filename = f"instiz_{year}-{month:02d}_sampled.csv"
            monthly_data.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"\n💾 저장: {filename}")
            
            # 다음 월로 넘어가기 전 대기
            time.sleep(5)
        
        # 전체 데이터 통합
        full_data = pd.concat(all_data, ignore_index=True)
        
        # 최종 저장
        final_filename = f"instiz_{year}_full_sampled.csv"
        full_data.to_csv(final_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n{'='*60}")
        print(f"🎉 전체 수집 완료!")
        print(f"총 게시물: {len(full_data):,}개")
        print(f"저장 파일: {final_filename}")
        print(f"{'='*60}")
        
        return full_data


# ============================================================================
# 사용 예시
# ============================================================================

if __name__ == "__main__":
    # 크롤러 초기화 (시간대당 1,250개씩)
    sampler = InstizTimeSampler(target_per_timerange=1250)
    
    # 옵션 1: 단일 월 수집
    # data = sampler.sample_month(2024, 1)
    
    # 옵션 2: 1년 전체 수집
    data = sampler.sample_year(2024)
    
    # 수집 결과 확인
    print("\n📊 수집 데이터 요약:")
    print(data.info())
    print("\n시간대별 분포:")
    print(data.groupby('timerange').size())
