from __future__ import annotations

import json
import logging
import os
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from .parsers import soupify, max_page_from_pagination, extract_list_items, extract_post
from .session import detect_login_required, SessionExpired
from .utils import (
    ensure_dir,
    extract_query_int,
    iter_months,
    sleep_jitter,
    to_month_str,
)
from urllib.parse import urljoin, urlencode


log = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    # 기본 사이트 베이스 URL (변경할 일 거의 없음)
    base: str = "https://www.instiz.net/"
    # 대상 보드 경로 (예: 'name' → https://www.instiz.net/name)
    board_path: str = "name"
    # 검색 파라미터: 카테고리/검색키/검색타입(기간 검색)
    category: int = 1
    stype: int = 9
    k: str = "기간탐색"
    # 페이지 시작 번호 (일반적으로 1)
    page_start: int = 1
    # 페이지 당 게시물 수(추정치). 동적 간격 계산에 사용됨
    posts_per_page: int = 20
    # 월별 목표 수집 게시물 수. 이 수에 도달하면 수집을 멈춥니다.
    target_posts_per_month: int = 50
    # 시간대별 목표 수집 게시물 수. 지정되지 않으면 target_posts_per_month를 시간대 수로 나눔
    target_posts_per_range: Optional[int] = None
    # 시간대 정의 (시작, 종료, 레이블). None이면 기본 4개 시간대 사용
    time_ranges: Optional[List[Tuple[str, str, str]]] = None
    # 요청 간 최소/최대 대기(초). 서버 부담 완화 목적
    min_delay: float = 3.0
    max_delay: float = 5.0
    # 요청 타임아웃(초) 및 재시도 횟수
    timeout: int = 15
    max_retries: int = 3
    # 파서 선택자 설정(JSON). 지정 시 사이트 구조 변경에 유연하게 대응
    selectors: Optional[Dict] = None
    # 고정 n-간격(예: 10 → 1,11,21,...) 강제. 지정 시 동적 간격 대신 이 값을 사용
    fixed_step: Optional[int] = None
    # 추가 옵션 (CLI 호환성)
    flush_every_posts: int = 0
    end_inclusive: bool = False
    end_at_2359: bool = False
    first_day_boundary: bool = False
    strict_month: bool = False
    skip_list_top_n: int = 0
    comments_count_from_parsed: bool = True
    comment_created_fallback: str = "blank"
    hash_fragment: Optional[str] = None
    sequential_pages: bool = True
    ascending: bool = True
    enforce_search_params: bool = False
    strict_url_encoding: bool = False
    # 빈 목록 페이지(삭제 등) 역방향 보정 시도 횟수와 간격(페이지 단위)
    empty_page_backtrack_limit: int = 3
    empty_page_backtrack_step: int = 10


class InstizCrawler:
    def __init__(
        self,
        session: requests.Session,
        cfg: CrawlConfig,
        output_dir: str,
        verbose: bool = False,
        dry_run: bool = False,
        skip_existing: bool = False,
        output_format: str = "jsonl",  # jsonl or csv
    ) -> None:
        """Instiz 크롤러

        - n-배수 샘플링: total_pages / P_needed 로 step 산출 또는 --fixed-step 사용
        - 인증: 주입된 쿠키 기반으로 요청, 로그인 페이지 신호 감지 시 SessionExpired 발생
        - 출력: JSONL 또는 CSV(게시글/댓글 분리)
        """
        self.s = session
        self.cfg = cfg
        self.output_dir = output_dir
        self.verbose = verbose
        self.dry_run = dry_run
        self.skip_existing = skip_existing
        self.output_format = output_format.lower()
        if not self.cfg.time_ranges:
            # 시간대 분할 대신 월 전체를 단일 구간으로 처리
            self.cfg.time_ranges = [(None, None, "전체")]

    def run_range(self, start: date, end: date, month_step: int = 1) -> None:
        ensure_dir(self.output_dir)
        for ms, me in iter_months(start, end, step=month_step):
            self.run_month(ms, me)

    def run_month(self, month_start: date, month_end: date) -> None:
        month_str = to_month_str(month_start)
        out_path_jsonl = os.path.join(self.output_dir, f"instiz_{month_str}.jsonl")
        out_csv = os.path.join(self.output_dir, f"instiz_{month_str}.csv")
        if self.verbose:
            target_desc = out_path_jsonl if self.output_format == "jsonl" else out_csv
            log.info(f"Month {month_str} → {target_desc}")
        if self.skip_existing and (
            (self.output_format == "jsonl" and os.path.exists(out_path_jsonl)) or
            (self.output_format == "csv" and os.path.exists(out_csv))
        ):
            if self.verbose:
                log.info("Skip existing outputs for this month")
            return
        if self.dry_run:
            log.info(self._month_info_preview(month_start, month_end))
            time_ranges = self.cfg.time_ranges or []
            target_per_range = self._resolve_target_per_range(len(time_ranges))
            for start_time, end_time, label in time_ranges:
                total_pages = self.detect_total_pages(
                    month_start,
                    month_end,
                    start_time=start_time,
                    end_time=end_time,
                )
                if total_pages is None or total_pages <= 0:
                    total_pages = 1
                pages = self._calculate_sampling_pages(total_pages, target_per_range)
                log.info(
                    f"[PLAN][{label}] total_pages={total_pages}, target={target_per_range}, pick={len(pages)}"
                )
                if pages:
                    sample_url = self.build_search_url(
                        month_start,
                        month_end,
                        pages[0],
                        start_time=start_time,
                        end_time=end_time,
                    )
                    log.info(f"[PLAN][{label}] sample_list_url={sample_url}")
                    preview = pages[:10]
                    tail = " ..." if len(pages) > 10 else ""
                    log.info(f"[PLAN][{label}] pages_preview={preview}{tail}")
            return

        seen_ids = set()
        records: List[Dict] = []
        time_ranges = self.cfg.time_ranges or []
        target_per_range = self._resolve_target_per_range(len(time_ranges))
        for start_time, end_time, label in time_ranges:
            total_pages = self.detect_total_pages(month_start, month_end, start_time=start_time, end_time=end_time)
            if total_pages is None or total_pages <= 0:
                total_pages = 1
            pages = self._calculate_sampling_pages(total_pages, target_per_range)
            if self.verbose:
                log.info(
                    f"[{label}] total_pages={total_pages}, target={target_per_range}, pick={len(pages)} pages"
                )

            collected_in_range = 0
            total_selected_pages = len(pages)
            for idx, p in enumerate(pages, 1):
                if collected_in_range >= target_per_range:
                    break
                list_url = self.build_search_url(
                    month_start, month_end, p, start_time=start_time, end_time=end_time
                )
                html = self.fetch_html(list_url)
                soup = soupify(html)
                items = extract_list_items(soup, selectors=self.cfg.selectors, board_path=self.cfg.board_path)
                # 빈 페이지(예: 마지막 페이지에 게시물 삭제) 예방: 역방향으로 보정 시도
                if not items and self.cfg.empty_page_backtrack_limit > 0:
                    attempts = 0
                    step = max(1, int(self.cfg.empty_page_backtrack_step or 1))
                    while not items and attempts < self.cfg.empty_page_backtrack_limit:
                        attempts += 1
                        bp = p - attempts * step
                        if bp < self.cfg.page_start:
                            break
                        list_url_bt = self.build_search_url(
                            month_start, month_end, bp, start_time=start_time, end_time=end_time
                        )
                        bt_html = self.fetch_html(list_url_bt)
                        bt_soup = soupify(bt_html)
                        bt_items = extract_list_items(bt_soup, selectors=self.cfg.selectors, board_path=self.cfg.board_path)
                        if bt_items:
                            items = bt_items
                            if self.verbose:
                                log.info(f"Backfilled empty list page p={p} -> p={bp} items={len(items)}")
                            break
                if self.cfg.skip_list_top_n:
                    items = items[self.cfg.skip_list_top_n :]
                if self.verbose:
                    log.info(f"[{idx}/{total_selected_pages}] page={p} items={len(items)}")
                for it in items:
                    if collected_in_range >= target_per_range:
                        break
                    post_url = self.normalize_url(it["url"])  # may be relative
                    pid = self.extract_post_id(post_url)
                    if pid is None:
                        pid = f"url:{post_url}"
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    post_html = self.fetch_html(post_url)
                    psoup = soupify(post_html)
                    pdata = extract_post(psoup, selectors=self.cfg.selectors)
                    list_comments_count = it.get("comments_count")
                    comments_count = (
                        list_comments_count
                        if list_comments_count is not None
                        else (
                            pdata.get("comments_count")
                            if pdata.get("comments_count") is not None
                            else len(pdata.get("comments") or [])
                        )
                    )
                    rec = {
                        "id": pid,
                        "url": post_url,
                        "title": pdata.get("title") or it.get("title") or "",
                        "body": pdata.get("body") or "",
                        "created_at": pdata.get("created_at") or "",
                        "likes": pdata.get("likes") if pdata.get("likes") is not None else 0,
                        "comments_count": comments_count,
                        "comments": pdata.get("comments") or [],
                        # 메타 컬럼 축소: fetched_at, month, timerange 및 timerange_* 제거
                    }
                    records.append(rec)
                    collected_in_range += 1
            if self.verbose:
                log.info(f"[{label}] collected={collected_in_range}")

        # Write outputs
        if self.output_format == "csv":
            self._write_csv_month(records, out_csv)
            if self.verbose:
                log.info(f"Saved {len(records)} posts → {out_csv}")
        else:
            with open(out_path_jsonl, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if self.verbose:
                log.info(f"Saved {len(records)} records → {out_path_jsonl}")

    def detect_total_pages(
        self,
        month_start: date,
        month_end: date,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Optional[int]:
        url = self.build_search_url(
            month_start,
            month_end,
            page=self.cfg.page_start,
            start_time=start_time,
            end_time=end_time,
        )
        html = self.fetch_html(url)
        soup = soupify(html)
        mp = max_page_from_pagination(soup)
        return mp

    def build_search_url(
        self,
        month_start: date,
        month_end: date,
        page: int,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> str:
        # Build: https://www.instiz.net/{board}?page=X&category=1&k=기간탐색&stype=9&starttime=YYYY/MM/DD+00:00&endtime=YYYY/MM/DD+00:00
        if start_time is None and end_time is None:
            start_param = month_start.strftime("%Y/%m/%d+00:00")
            end_param = month_end.strftime("%Y/%m/%d+00:00")
        else:
            s_time = self._parse_time_str(start_time or "00:00")
            e_time = self._parse_time_str(end_time or "23:59")
            last_day = month_end - timedelta(days=1)
            if last_day < month_start:
                last_day = month_start
            start_dt = datetime.combine(month_start, s_time)
            end_dt = datetime.combine(last_day, e_time)
            start_param = start_dt.strftime("%Y/%m/%d+%H:%M")
            end_param = end_dt.strftime("%Y/%m/%d+%H:%M")
        params = {
            "page": page,
            "category": self.cfg.category,
            "k": self.cfg.k,
            "stype": self.cfg.stype,
            "starttime": start_param,
            "endtime": end_param,
        }
        base = self.normalize_url(f"/{self.cfg.board_path}")
        url = base + "?" + urlencode(params, doseq=True, safe=":/+\n ")
        if self.cfg.hash_fragment:
            url = f"{url}#{self.cfg.hash_fragment}"
        return url

    def normalize_url(self, href: str) -> str:
        return urljoin(self.cfg.base, href)

    def fetch_html(self, url: str) -> str:
        tries = 0
        last_exc: Optional[Exception] = None
        while tries <= self.cfg.max_retries:
            tries += 1
            try:
                if self.verbose:
                    log.debug(f"GET {url}")
                resp = self.s.get(url, timeout=self.cfg.timeout)
                text = resp.text or ""
                if detect_login_required(text, resp.url):
                    raise SessionExpired(f"Login required or session expired at {resp.url}")
                sleep_jitter(self.cfg.min_delay, self.cfg.max_delay)
                return text
            except SessionExpired:
                raise
            except Exception as e:
                last_exc = e
                sleep_jitter(self.cfg.min_delay, self.cfg.max_delay)
        if last_exc:
            raise last_exc
        return ""

    def extract_post_id(self, url: str) -> Optional[str]:
        for key in ("no", "id", "article", "artno"):
            v = extract_query_int(url, key)
            if v is not None:
                return f"{key}:{v}"
        # 예: 경로가 /name/123456 형태인 경우
        m = None
        try:
            m = __import__("re").search(rf"/{self.cfg.board_path}/(\d+)", url)
        except Exception:
            pass
        if m:
            return f"id:{m.group(1)}"
        return None

    def _resolve_target_per_range(self, range_count: int) -> int:
        if range_count <= 0:
            return max(1, self.cfg.target_posts_per_month)
        if self.cfg.target_posts_per_range and self.cfg.target_posts_per_range > 0:
            return max(1, self.cfg.target_posts_per_range)
        base = max(1, self.cfg.target_posts_per_month)
        return max(1, -(-base // range_count))

    def _calculate_sampling_pages(self, total_pages: int, target_posts: int) -> List[int]:
        total_pages = max(1, total_pages)
        posts_per_page = max(1, self.cfg.posts_per_page)
        pages_needed = max(1, -(-target_posts // posts_per_page))
        if self.cfg.fixed_step and self.cfg.fixed_step > 0:
            return self._build_pages(
                self.cfg.page_start,
                total_pages,
                self.cfg.fixed_step,
                pages_needed,
            )
        if pages_needed >= total_pages:
            return list(range(self.cfg.page_start, total_pages + 1))
        step = max(1, round(total_pages / pages_needed))
        return self._build_pages(self.cfg.page_start, total_pages, step, pages_needed)

    def _build_pages(self, start: int, total: int, step: int, pages_needed: int) -> List[int]:
        pages: List[int] = []
        p = start
        while p <= total and len(pages) < pages_needed:
            pages.append(p)
            p += step
        if pages and pages[-1] != total:
            pages.append(total)
        pages = sorted(set([x for x in pages if 1 <= x <= total]))
        return pages

    def _month_info_preview(self, month_start: date, month_end: date) -> str:
        url = self.build_search_url(month_start, month_end, self.cfg.page_start)
        return f"[DRY-RUN] {to_month_str(month_start)} first URL → {url}"

    def _write_csv_month(self, records: List[Dict], csv_path: str) -> None:
        headers = [
            "id",
            "url",
            "title",
            "body",
            "created_at",
            "likes",
            "comments_count",
            # 축소: fetched_at, month, timerange, timerange_* 제거
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as pf:
            pw = csv.writer(pf)
            pw.writerow(headers)
            for r in records:
                pw.writerow([
                    r.get("id", ""),
                    r.get("url", ""),
                    r.get("title", ""),
                    r.get("body", ""),
                    r.get("created_at", ""),
                    r.get("likes", 0),
                    r.get("comments_count", 0),
                ])

    def _parse_time_str(self, value: str) -> datetime.time:
        try:
            return datetime.strptime(value, "%H:%M").time()
        except Exception:
            return datetime.strptime("00:00", "%H:%M").time()
