# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

인스티즈 커뮤니티 크롤러로 한국 포럼 instiz.net에서 게시글과 댓글을 수집하는 도구입니다. 지능적인 샘플링과 세션 관리를 구현하여 하루 4개 시간대(새벽, 오전, 오후, 저녁)로 나누어 균형잡힌 데이터 수집을 수행합니다.

## 주요 명령어

**의존성 설치:**
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # 개발용
```

**테스트 실행:**
```bash
pytest -q                           # 빠른 테스트
pytest tests/ -v                    # 상세 테스트  
pytest tests/test_crawler_unit.py   # 특정 테스트 파일
```

**크롤러 테스트 (실제 요청 없음):**
```bash
python -m instiz_crawler.cli --dry-run --verbose --start 2024-01-01 --end 2024-01-31
```

**크롤러 실행 (유효한 쿠키 필요):**
```bash
python -m instiz_crawler.cli \
  --cookies-json cookies.json \
  --start 2024-01-01 --end 2024-01-31 \
  --target-posts-per-month 1000
```

## 아키텍처

### 핵심 컴포넌트

**`instiz_crawler/cli.py`**: 69개 이상의 명령줄 옵션을 가진 진입점, 프리셋 시스템(research/sampling/none), 쿠키 관리(string/json/auto 모드)

**`instiz_crawler/crawler.py`**: `InstizCrawler` 클래스를 포함한 메인 크롤링 엔진:
- 시간대별 샘플링: 균등한 시간 분포를 위한 4개 일일 시간대
- 동적 페이지 샘플링: `step = max(1, round(total_pages / pages_needed))`
- 월별 파일 조직화와 CSV/JSONL 출력

**`instiz_crawler/parsers.py`**: `selectors.json`의 설정 가능한 CSS 선택자를 사용한 BeautifulSoup 기반 HTML 파싱, 다단계 추출(목록 → 게시글 → 댓글), 폴백 메커니즘

**`instiz_crawler/session.py`**: 로그인 감지 휴리스틱과 `SessionExpired` 예외 처리를 포함한 쿠키 기반 인증

**`instiz_crawler/utils.py`**: 날짜/시간 유틸리티, 네트워크 헬퍼(지터 지연), 파일 시스템 작업

### 설정 시스템

**`selectors.json`**: 코드 변경 없이 DOM 적응을 위한 외부화된 CSS 선택자:
```json
{
  "list_item_link": ["td.t_left > a"],
  "post_body_strict": ["td#content_td p", "td#content_td .text"],
  "comments": {"item": "table[node='_44'] tr", "content": "td"}
}
```

**쿠키 관리**: 브라우저 쿠키 문자열 또는 도메인/경로 제어가 가능한 구조화된 JSON 형식 지원

**프리셋 시스템**: 다양한 사용 사례에 최적화된 매개변수를 가진 사전 정의된 설정(research/sampling)

### 핵심 알고리즘

**시간대별 샘플링**: 각 월을 새벽(00-06), 오전(06-12), 오후(12-18), 저녁(18-24)으로 나누어 균형잡힌 시간적 표현

**동적 페이지 샘플링**: 목표 게시글 수를 기반으로 최적의 페이지 간격 계산, 1, 1+step, 1+2×step과 같이 페이지 방문하여 월별 변동에 관계없이 일관된 샘플 크기 보장

**다층 파싱**: 폴백 옵션을 가진 기본 선택자, 엄격한 vs 느슨한 파싱 모드, 콘텐츠 정리를 위한 제외 구역

## 테스트

실제 네트워크 요청을 피하기 위해 포괄적인 모킹을 사용하는 pytest 기반:
- URL 구축 및 매개변수 인코딩 (`test_crawler_unit.py`)
- 현실적인 픽스처를 사용한 HTML 파싱 (`test_parsers.py`) 
- 세션 관리 및 로그인 감지 (`test_session.py`)
- 모킹된 응답을 사용한 통합 워크플로우 (`test_integration_mocked.py`)

모킹 패턴은 현실적인 HTML 픽스처로 HTTP 응답을 시뮬레이션하기 위해 `FakeSession`과 `FakeResp` 클래스를 사용합니다.

## 출력 형식

**게시글**: `data/instiz_YYYY-MM_posts.csv` 컬럼: id, url, title, body, created_at, likes, comments_count

**댓글**: `data/instiz_YYYY-MM_comments.csv` 컬럼: post_id, index, content, created_at

둘 다 점진적 쓰기와 플러시 간격을 가진 CSV(기본값) 및 JSONL 형식을 지원합니다.

## 주요 의존성

- **requests**: 세션 관리를 포함한 HTTP 클라이언트
- **beautifulsoup4 + lxml**: HTML 파싱 및 DOM 탐색
- **pytest**: 테스트 프레임워크

## 개발 참고사항

- **설정 우선 접근법**: 쉬운 사이트 적응을 위해 선택자 외부화
- **세션 만료 처리**: 우아한 실패와 함께 자동 로그인 페이지 감지
- **윤리적 크롤링**: 내장된 지연, 재시도 제한, robots.txt 준수
- **품질 보증**: 중복 제거, 텍스트 정리, 콘텐츠 검증