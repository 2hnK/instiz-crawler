from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date
from typing import Optional

from .crawler import CrawlConfig, InstizCrawler
from .session import build_session, SessionExpired
from .utils import parse_date


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="인스티즈 월별 크롤러 (n-간격/연속 방문 지원)")
    p.add_argument("--start", required=True, help="시작 날짜 YYYY-MM-DD")
    p.add_argument("--end", required=True, help="종료 날짜 YYYY-MM-DD")
    p.add_argument("--month-step", type=int, default=1, help="월 간격(1=매월)")

    p.add_argument("--board-path", default="name", help="보드 경로(예: name)")
    p.add_argument("--category", type=int, default=1, help="카테고리 번호")
    p.add_argument("--stype", type=int, default=9, help="검색 타입(기간탐색)")
    p.add_argument("--k", default="기간탐색", help="키워드/검색키")

    p.add_argument("--target-posts-per-month", type=int, default=45, help="월별 목표 게시글 수")
    p.add_argument("--posts-per-page", type=int, default=20, help="페이지당 게시글 수(추정)")
    p.add_argument("--page-start", type=int, default=1, help="목록 시작 페이지")
    p.add_argument("--fixed-step", type=int, default=None, help="고정 간격 방문(예: 10 → 1,11,21..)")
    p.add_argument("--end-inclusive-last-day", action="store_true", help="endtime을 해당 월 마지막날 00:00으로 설정")
    p.add_argument("--end-2359", action="store_true", help="endtime을 해당 월 마지막날 23:59로 설정")
    p.add_argument("--skip-list-top-n", type=int, default=0, help="각 페이지 상단 인기/고정글 N개 건너뛰기")
    p.add_argument("--flush-every-posts", type=int, default=20, help="게시글 N개마다 CSV를 즉시 갱신(0=비활성)")
    p.add_argument("--first-day-boundary", action="store_true", help="start/end가 항상 1일 경계면 end(다음달 1일)를 배제해 한 달만 수집")
    # default None: 미지정 시 크롤러 기본(True) 유지
    p.add_argument("--comments-count-from-parsed", action="store_true", default=None,
                   help="파싱된 댓글 개수(len)로 comments_count 사용")
    p.add_argument("--comment-created-fallback", choices=["blank", "now", "post"], default="blank",
                   help="댓글 작성시각 누락 시 대체: blank/now(수집시각)/post(게시글 시각)")

    p.add_argument("--output-dir", default="data", help="출력 디렉터리")
    p.add_argument("--output-format", choices=["jsonl", "csv"], default="csv", help="출력 포맷")
    p.add_argument("--skip-existing", action="store_true", help="해당 월 결과 파일이 있으면 건너뛰기")
    p.add_argument("--use-greentop", action="store_true", default=True, help="목록 URL 끝에 #greentop 앵커 추가(기본값: 사용)")
    p.add_argument("--preset", choices=["research", "sampling", "none"], default="none",
                   help="권장 프리셋 적용(research: 일반 수집, sampling: 동적 샘플링 단축)")
    p.add_argument("--sequential-pages", action="store_true", default=True, help="목록 페이지를 1..N 순서로 순차 방문")
    p.add_argument("--dynamic-sampling", action="store_true", help="동적 간격 샘플링(1,1+step,1+2*step,...) 사용")
    p.add_argument("--ascending", action="store_true", help="오래된→최신 순서로 처리(페이지/항목 역순)")
    p.add_argument("--enforce-search-filter", action="store_true", help="검색 파라미터(k/starttime/endtime/stype)가 유지된 링크만 수집")
    p.add_argument("--strict-url-encoding", action="store_true", help="쿼리 문자열에서 /, : 도 인코딩(%2F, %3A) 처리")
    p.add_argument("--strict-month", action="store_true", help="created_at이 월 범위를 벗어나면 게시글 제외")

    p.add_argument("--cookies-json", help="cookies.json 경로", default="cookies.json")
    p.add_argument("--cookie-string", help="브라우저 Cookie 헤더 문자열", default=None)
    p.add_argument("--cookie-mode", choices=["auto", "string", "json"], default="auto",
                   help="cookie 제공 방식: auto/string/json")
    p.add_argument("--http-proxy", default=None, help="HTTP 프록시 URL")
    p.add_argument("--https-proxy", default=None, help="HTTPS 프록시 URL")

    p.add_argument("--min-delay", type=float, default=3.0, help="요청 간 최소 대기(초)")
    p.add_argument("--max-delay", type=float, default=7.0, help="요청 간 최대 대기(초)")
    p.add_argument("--timeout", type=int, default=15, help="요청 타임아웃(초)")
    p.add_argument("--max-retries", type=int, default=3, help="최대 재시도 횟수")

    p.add_argument("--selectors", help="선택자 설정 JSON 경로", default="selectors.json")
    p.add_argument("--verbose", action="store_true",default=True, help="자세한 로그 출력")
    p.add_argument("--dry-run", action="store_true", help="URL/계획만 출력(요청 최소화)")
    return p


def load_selectors(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = build_arg_parser().parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    sdate = parse_date(args.start)
    edate = parse_date(args.end)

    # Apply preset defaults (only when user didn't explicitly pass alternatives)
    if args.preset == "sampling":
        # selectors.json / cookies.json 자동 적용
        if not args.selectors and os.path.exists("selectors.json"):
            args.selectors = "selectors.json"
        if not args.cookies_json and not args.cookie_string and os.path.exists("cookies.json"):
            args.cookies_json = "cookies.json"
        # 기간/URL/필터 관련 권장값
        if not args.first_day_boundary:
            args.first_day_boundary = True
        if not args.end_2359:
            args.end_2359 = True
        if not args.enforce_search_filter:
            args.enforce_search_filter = True
        if not args.strict_url_encoding:
            args.strict_url_encoding = True
        # 동적 샘플링: sequential-pages 대신 dynamic-sampling 사용
        args.sequential_pages = False
        args.dynamic_sampling = True
        # 댓글수는 파싱된 길이 기준
        if args.comments_count_from_parsed is None:
            args.comments_count_from_parsed = True
        # 상단 스킵은 0으로(실질 p와 step 불일치 방지)
        args.skip_list_top_n = 0
        if not getattr(args, "enforce_search_filter", False):
            args.enforce_search_filter = True
        if not getattr(args, "strict_url_encoding", False):
            args.strict_url_encoding = True

    selectors = load_selectors(args.selectors)
    # 동적 샘플링 플래그가 켜지면 순차 방문을 끕니다.
    if args.dynamic_sampling:
        args.sequential_pages = False

    cfg = CrawlConfig(
        board_path=args.board_path,
        category=args.category,
        stype=args.stype,
        k=args.k,
        page_start=args.page_start,
        posts_per_page=args.posts_per_page,
        target_posts_per_month=args.target_posts_per_month,
        flush_every_posts=args.flush_every_posts,
        end_inclusive=args.end_inclusive_last_day,
        end_at_2359=args.end_2359,
        first_day_boundary=args.first_day_boundary,
        min_delay=args.min_delay,
        max_delay=args.max_delay,
        timeout=args.timeout,
        max_retries=args.max_retries,
        selectors=selectors,
        fixed_step=args.fixed_step,
        strict_month=args.strict_month,
        skip_list_top_n=args.skip_list_top_n,
        comments_count_from_parsed=(args.comments_count_from_parsed if args.comments_count_from_parsed is not None else True),
        comment_created_fallback=args.comment_created_fallback,
        hash_fragment=("greentop" if args.use_greentop else None),
        sequential_pages=args.sequential_pages,
        ascending=args.ascending if hasattr(args, "ascending") else True,
        enforce_search_params=args.enforce_search_filter if hasattr(args, "enforce_search_filter") else False,
        strict_url_encoding=args.strict_url_encoding if hasattr(args, "strict_url_encoding") else False,
    )

    # 쿠키 모드 검증: 사용자가 원하는 제공 방식을 명시적으로 선택 가능
    cookie_string = args.cookie_string
    cookies_json = args.cookies_json
    if args.cookie_mode == "string":
        if not cookie_string:
            raise SystemExit("--cookie-mode string 인 경우 --cookie-string 을 제공해야 합니다.")
        cookies_json = None
    elif args.cookie_mode == "json":
        if not cookies_json:
            raise SystemExit("--cookie-mode json 인 경우 --cookies-json 을 제공해야 합니다.")
        cookie_string = None
    else:
        # auto: 명시적 충돌 방지. 둘 다 제공 시 JSON을 우선하거나 한쪽만 사용.
        if cookies_json and cookie_string:
            # JSON 우선
            cookie_string = None

    sess = build_session(cookie_string=cookie_string, cookies_json_path=cookies_json)
    if args.http_proxy or args.https_proxy:
        sess.proxies = {}
        if args.http_proxy:
            sess.proxies["http"] = args.http_proxy
        if args.https_proxy:
            sess.proxies["https"] = args.https_proxy
    crawler = InstizCrawler(
        sess,
        cfg,
        output_dir=args.output_dir,
        verbose=args.verbose,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        output_format=args.output_format,
    )
    try:
        crawler.run_range(sdate, edate, month_step=args.month_step)
    except SessionExpired as e:
        print("[ERROR] Session expired or login required. Please refresh cookies and re-run.")
        print(str(e))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
