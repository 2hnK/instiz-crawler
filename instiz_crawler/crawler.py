from __future__ import annotations

import json
import logging
import os
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Tuple
import re

import requests
from bs4 import BeautifulSoup

from .parsers import (
    soupify,
    max_page_from_pagination,
    extract_list_items,
    extract_post,
    extract_comments,
)
from .session import detect_login_required, SessionExpired
from .utils import (
    ensure_dir,
    extract_query_int,
    parse_datetime_loose,
    iter_months,
    sleep_jitter,
    to_month_str,
)
from .utils import iso_now, format_ampm_str
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
    target_posts_per_month: int = 400
    # 요청 간 최소/최대 대기(초). 서버 부담 완화 목적
    min_delay: float = 2.0
    max_delay: float = 10.0
    # 요청 타임아웃(초) 및 재시도 횟수
    timeout: int = 15
    max_retries: int = 2
    # 파서 선택자 설정(JSON). 지정 시 사이트 구조 변경에 유연하게 대응
    selectors: Optional[Dict] = None
    # 고정 n-간격(예: 10 → 1,11,21,...) 강제. 지정 시 동적 간격 대신 이 값을 사용
    fixed_step: Optional[int] = None
    # 생성일이 요청 월 범위를 벗어나면 해당 게시글을 제외
    strict_month: bool = False
    # 각 목록 페이지 상단 고정/인기글 등 N개를 건너뛰기
    skip_list_top_n: int = 0
    # endtime을 해당 월의 마지막 날 00:00으로 설정(기본: 다음달 1일 00:00)
    end_inclusive: bool = False
    # URL 해시 프래그먼트(예: 'greentop')
    hash_fragment: Optional[str] = None
    # 댓글 개수는 항상 파싱된 길이(len(comments))로 결정
    comments_count_from_parsed: bool = True
    # 댓글 created_at 결측 시 대체 방식: blank|now|post
    comment_created_fallback: str = "blank"
    # 목록 페이지를 순차적으로 1..N 방문할지 여부(기본: 순차 방문)
    sequential_pages: bool = True
    # endtime을 항상 해당 월 마지막 날 23:59로 보낼지 여부
    end_at_2359: bool = False
    # 오래된 글부터 수집(오름차순)할지 여부. 기본은 최신→과거(내림차순)
    ascending: bool = False
    # 검색 결과 URL의 쿼리 파라미터(예: k/starttime/endtime 등)가 포함된 링크만 허용
    enforce_search_params: bool = False
    # URL 쿼리 인코딩을 엄격 모드로(슬래시/콜론까지 인코딩)
    strict_url_encoding: bool = False
    # 수집 중 CSV를 부분 갱신하는 주기(게시글 N개마다). 0이면 비활성
    flush_every_posts: int = 20
    # 시작/종료가 매월 1일 경계인 경우 end(다음달 1일)를 하루 줄여 해당 월만 포함
    first_day_boundary: bool = False


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

    def run_range(self, start: date, end: date, month_step: int = 1) -> None:
        ensure_dir(self.output_dir)
        # 첫날 경계 모드: start/end가 모두 1일이면 end를 하루 줄여 해당 월만 포함
        adj_end = end
        if (
            getattr(self.cfg, "first_day_boundary", False)
            and start.day == 1 and end.day == 1 and start < end
        ):
            from datetime import timedelta
            adj_end = end - timedelta(days=1)
        for ms, me in iter_months(start, adj_end, step=month_step):
            self.run_month(ms, me)

    def run_month(self, month_start: date, month_end: date) -> None:
        month_str = f"{month_start.year:04d}-{month_start.month:02d}"
        out_path_jsonl = os.path.join(self.output_dir, f"instiz_{month_str}.jsonl")
        out_posts_csv = os.path.join(self.output_dir, f"instiz_{month_str}_posts.csv")
        out_comments_csv = os.path.join(self.output_dir, f"instiz_{month_str}_comments.csv")
        if self.verbose:
            target_desc = out_path_jsonl if self.output_format == "jsonl" else f"{out_posts_csv} & {out_comments_csv}"
            log.info(f"Month {month_str} → {target_desc}")
        if self.skip_existing and (
            (self.output_format == "jsonl" and os.path.exists(out_path_jsonl)) or
            (self.output_format == "csv" and os.path.exists(out_posts_csv) and os.path.exists(out_comments_csv))
        ):
            if self.verbose:
                log.info("Skip existing outputs for this month")
            return
        if self.dry_run:
            log.info(self._month_info_preview(month_start, month_end))
            return

        # Determine total pages for this month
        total_pages = self.detect_total_pages(month_start, month_end)
        if total_pages is None:
            total_pages = 1
        # Build page plan
        if self.cfg.sequential_pages:
            if self.cfg.ascending:
                pages = list(range(total_pages, self.cfg.page_start - 1, -1))
                if self.verbose:
                    log.info(f"Total pages={total_pages}, sequential visit descending pages for ascending chronology")
            else:
                pages = list(range(self.cfg.page_start, total_pages + 1))
                if self.verbose:
                    log.info(f"Total pages={total_pages}, sequential visit from {self.cfg.page_start}")
        else:
            # Compute step (fixed or from target posts)
            if self.cfg.fixed_step and self.cfg.fixed_step > 0:
                step = self.cfg.fixed_step
                # approximate pages needed
                pages_needed = max(1, (total_pages + step - 1) // step)
            else:
                pages_needed = max(1, -(-self.cfg.target_posts_per_month // max(1, self.cfg.posts_per_page)))
                step = max(1, round(total_pages / pages_needed))
            pages = self._build_pages(self.cfg.page_start, total_pages, step, pages_needed)
            if self.cfg.ascending:
                pages = list(reversed(pages))
            if self.verbose:
                log.info(f"Total pages={total_pages}, need={pages_needed}, step={step}, pick={len(pages)} pages")

        seen_ids = set()
        records: List[Dict] = []
        buffer: List[Dict] = []
        wrote_header = False
        for p in pages:
            list_url = self.build_search_url(month_start, month_end, p)
            if self.verbose:
                log.info(f"page={p} url={list_url}")
            html = self.fetch_html(list_url)
            soup = soupify(html)
            items = extract_list_items(soup, selectors=self.cfg.selectors, board_path=self.cfg.board_path)
            # 검색 결과 필터: 쿼리 파라미터 없는 최신/사이드 링크 배제
            if self.cfg.enforce_search_params:
                before = len(items)
                items = [it for it in items if self._is_search_result_link(it.get("url", ""))]
                if self.verbose and before != len(items):
                    log.info(f"page={p} filtered_by_search_params kept={len(items)} removed={before-len(items)}")
            if self.cfg.ascending:
                items = list(reversed(items))
            if self.verbose:
                log.info(f"page={p} items={len(items)}")
            # 상단 인기/고정글 스킵
            if self.cfg.skip_list_top_n > 0 and items:
                if self.cfg.ascending:
                    # 원래 상단 N개는 역순에서 리스트의 끝에 위치
                    if len(items) > self.cfg.skip_list_top_n:
                        items = items[:-self.cfg.skip_list_top_n]
                    else:
                        items = []
                else:
                    items = items[self.cfg.skip_list_top_n:]
            for it in items:
                post_url = self.normalize_url(it["url"])  # may be relative
                pid = self.extract_post_id(post_url)
                if pid is None:
                    # 게시글 식별자가 없는 링크(페이지/JS/프로필 등)는 스킵
                    if self.verbose:
                        log.info(f"skip non-post link url={post_url}")
                    continue
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                post_html = self.fetch_html(post_url)
                psoup = soupify(post_html)
                pdata = extract_post(psoup, selectors=self.cfg.selectors)
                # 본문이 비거나 의미 없으면 body를 null(None)로 유지
                if self.verbose and (not (pdata.get("body") or "").strip()):
                    log.info(f"Null body set id={pid}")
                # Collect comments including paginated ones
                all_comments = pdata.get("comments") or []
                more_comments = self._collect_more_comments(post_url, pid, psoup, total_comments=pdata.get("comments_count"))
                if more_comments:
                    all_comments.extend(more_comments)
                # 중복 제거 (content, created_at 기준)
                all_comments = self._dedupe_comments(all_comments)
                # Validate created_at within requested month range
                created_val = (pdata.get("created_at") or "").strip()
                created_val = self._validate_created_month(created_val, month_start, month_end)
                if self.cfg.strict_month and not created_val:
                    if self.verbose:
                        log.info("Skip post due to created_at outside month or missing")
                    continue
                rec = {
                    "id": pid,
                    "url": post_url,
                    "title": pdata.get("title") or it.get("title") or "",
                    "body": pdata.get("body") or "",
                    "created_at": created_val,
                    "likes": pdata.get("likes") if pdata.get("likes") is not None else 0,
                    "comments_count": (
                        len(all_comments) if self.cfg.comments_count_from_parsed else (
                            pdata.get("comments_count") if pdata.get("comments_count") is not None else len(all_comments)
                        )
                    ),
                    "comments": all_comments,
                }
                records.append(rec)
                buffer.append(rec)
                if self.verbose:
                    try:
                        cnum = len(all_comments)
                    except Exception:
                        cnum = 0
                    log.info(
                        f"Scraped post {len(records)}/{self.cfg.target_posts_per_month} id={pid} comments={cnum}"
                    )
                # Flush batch if needed
                if (
                    self.output_format == "csv"
                    and self.cfg.flush_every_posts and self.cfg.flush_every_posts > 0
                    and len(buffer) >= self.cfg.flush_every_posts
                ):
                    self._append_csv_batch(buffer, out_posts_csv, out_comments_csv)
                    buffer = []
                if len(records) >= self.cfg.target_posts_per_month:
                    break
            if len(records) >= self.cfg.target_posts_per_month:
                break

        # Write outputs
        if self.output_format == "csv":
            # Flush any remaining buffer
            if buffer:
                self._append_csv_batch(buffer, out_posts_csv, out_comments_csv)
                buffer = []
            # If no incremental flush configured, ensure file contains all
            if not self.cfg.flush_every_posts:
                self._write_csv_month(records, out_posts_csv, out_comments_csv)
            if self.verbose:
                log.info(f"Saved {len(records)} posts → {out_posts_csv}")
                # comments count
                ccount = sum(len(r.get("comments", [])) for r in records)
                log.info(f"Saved {ccount} comments → {out_comments_csv}")
        else:
            with open(out_path_jsonl, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if self.verbose:
                log.info(f"Saved {len(records)} records → {out_path_jsonl}")

    def detect_total_pages(self, month_start: date, month_end: date) -> Optional[int]:
        url = self.build_search_url(month_start, month_end, page=self.cfg.page_start)
        html = self.fetch_html(url)
        soup = soupify(html)
        mp = max_page_from_pagination(soup, selectors=self.cfg.selectors)
        return mp

    def build_search_url(self, month_start: date, month_end: date, page: int) -> str:
        # Build: https://www.instiz.net/{board}?page=X&category=1&k=기간탐색&stype=9&starttime=YYYY/MM/DD+00:00&endtime=YYYY/MM/DD+HH:MM
        # endtime 모드: end_at_2359(해당월 마지막날 23:59), inclusive(해당월 마지막날 00:00), 기본(다음달 1일 00:00)
        if self.cfg.end_at_2359:
            end_for_query = month_end - timedelta(days=1)
            end_time_str = end_for_query.strftime("%Y/%m/%d+23:59")
        else:
            end_for_query = month_end - timedelta(days=1) if self.cfg.end_inclusive else month_end
            end_time_str = end_for_query.strftime("%Y/%m/%d+00:00")
        params = {
            "page": page,
            "category": self.cfg.category,
            "k": self.cfg.k,
            "stype": self.cfg.stype,
            "starttime": month_start.strftime("%Y/%m/%d+00:00"),
            "endtime": end_time_str,
        }
        base = self.normalize_url(f"/{self.cfg.board_path}")
        safe_chars = "+" if self.cfg.strict_url_encoding else ":/+\n "
        url = base + "?" + urlencode(params, doseq=True, safe=safe_chars)
        if self.cfg.hash_fragment:
            url += f"#{self.cfg.hash_fragment}"
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

    def _is_search_result_link(self, href: str) -> bool:
        try:
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(href).query)
            # 기간탐색 파라미터 유무로 판단
            keys = set(k.lower() for k in q.keys())
            return bool(keys & {"k", "starttime", "endtime", "stype"})
        except Exception:
            return False

    def _write_csv_month(self, records: List[Dict], posts_path: str, comments_path: str) -> None:
        # Posts CSV
        with open(posts_path, "w", encoding="utf-8", newline="") as pf:
            pw = csv.writer(pf)
            pw.writerow(["id", "url", "title", "body", "created_at", "likes", "comments_count"])  # header
            for r in records:
                pw.writerow(self._format_post_row(r))
        # Comments CSV
        with open(comments_path, "w", encoding="utf-8", newline="") as cf:
            cw = csv.writer(cf)
            cw.writerow(["post_id", "index", "content", "created_at"])  # header
            for r in records:
                for row in self._iter_comment_rows(r):
                    cw.writerow(row)

    def _append_csv_batch(self, batch: List[Dict], posts_path: str, comments_path: str) -> None:
        # Append posts
        need_header_posts = not os.path.exists(posts_path)
        with open(posts_path, "a", encoding="utf-8", newline="") as pf:
            pw = csv.writer(pf)
            if need_header_posts:
                pw.writerow(["id", "url", "title", "body", "created_at", "likes", "comments_count"])  # header
            for r in batch:
                pw.writerow(self._format_post_row(r))
        # Append comments
        need_header_comments = not os.path.exists(comments_path)
        with open(comments_path, "a", encoding="utf-8", newline="") as cf:
            cw = csv.writer(cf)
            if need_header_comments:
                cw.writerow(["post_id", "index", "content", "created_at"])  # header
            for r in batch:
                for row in self._iter_comment_rows(r):
                    cw.writerow(row)

    def _format_post_row(self, r: Dict) -> List:
        created_fmt = format_ampm_str(r.get("created_at", ""))
        body_out = r.get("body")
        if body_out is None:
            body_out = ""
        return [
            r.get("id", ""),
            r.get("url", ""),
            r.get("title", ""),
            body_out,
            created_fmt,
            r.get("likes", 0),
            r.get("comments_count", 0),
        ]

    def _iter_comment_rows(self, r: Dict):
        post_id = r.get("id", "")
        comments = r.get("comments") or []
        for idx, c in enumerate(comments):
            c_created = (c.get("created_at") if isinstance(c, dict) else "") or ""
            if not c_created:
                if self.cfg.comment_created_fallback == "now":
                    c_created = iso_now()
                elif self.cfg.comment_created_fallback == "post":
                    c_created = r.get("created_at") or ""
            c_created_fmt = format_ampm_str(c_created)
            yield [
                post_id,
                idx,
                (c.get("content") if isinstance(c, dict) else str(c)),
                c_created_fmt,
            ]

    def _dedupe_comments(self, comments: List[Dict]) -> List[Dict]:
        seen = set()
        out: List[Dict] = []
        for c in comments:
            if isinstance(c, dict):
                key = (c.get("content") or "", c.get("created_at") or "")
            else:
                key = (str(c), "")
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _validate_created_month(self, created_at: str, month_start: date, month_end: date) -> str:
        dt = parse_datetime_loose(created_at)
        if not dt:
            return ""
        d = dt.date()
        if month_start <= d < month_end:
            return created_at
        return ""

    def _post_numeric_id(self, pid: Optional[str], url: str) -> Optional[str]:
        if pid and ":" in pid:
            tail = pid.split(":", 1)[1]
            if tail.isdigit():
                return tail
        try:
            m = re.search(rf"/{re.escape(self.cfg.board_path)}/(\d+)", url)
            if m:
                return m.group(1)
        except Exception:
            pass
        for key in ("no", "id", "article", "artno"):
            v = extract_query_int(url, key)
            if v is not None:
                return str(v)
        return None

    def _scan_comment_page_links(self, soup: BeautifulSoup, numeric_id: Optional[str], total_comments: Optional[int] = None) -> Dict[int, str]:
        out: Dict[int, str] = {}
        if not numeric_id:
            return out
        pat_path = re.compile(rf"/{re.escape(self.cfg.board_path)}/{re.escape(numeric_id)}(?:[^\d]|$)")
        pat_q = re.compile(rf"/{re.escape(self.cfg.board_path)}[^\s]*[?&](?:no|id|article|artno)={re.escape(numeric_id)}\b", re.I)
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "page=" not in href:
                continue
            if not (pat_path.search(href) or pat_q.search(href)):
                continue
            pn = extract_query_int(href, "page")
            if pn and pn > 1:
                out[pn] = self.normalize_url(href)
        # 추가: javascript:ajax(...) 형태의 댓글 페이저 처리 (selectors 기반 URL 템플릿 필요)
        js_links = self._scan_comment_js_pager(soup, numeric_id, total_comments=total_comments)
        out.update({k: v for k, v in js_links.items() if k not in out})
        return out

    def _scan_comment_js_pager(self, soup: BeautifulSoup, numeric_id: Optional[str], total_comments: Optional[int] = None) -> Dict[int, str]:
        out: Dict[int, str] = {}
        sel = (self.cfg.selectors or {}).get("comment_pager") if self.cfg.selectors else None
        if not sel or not isinstance(sel, dict):
            return out
        anchors = sel.get("anchors") or "a[href^='javascript:ajax']"
        js_pattern = sel.get("js_pattern") or r"javascript:ajax\s*\(\s*'(?P<board>[^']+)'\s*,\s*(?P<id>\d+)\s*,\s*(?P<page>\d+)"
        url_template = sel.get("url_template")  # 예: "{base}/{board}?no={id}&page={page}&cmt={cmt}"
        if not url_template:
            return out
        try:
            nodes = soup.select(anchors)
        except Exception:
            nodes = []
        cre = re.compile(js_pattern)
        for a in nodes:
            href = a.get("href") or ""
            m = cre.search(href)
            if not m:
                continue
            board = m.groupdict().get("board") or self.cfg.board_path
            pid = m.groupdict().get("id")
            page = m.groupdict().get("page")
            if not page:
                continue
            try:
                pn = int(page)
            except Exception:
                continue
            if pn <= 1:
                continue
            # numeric_id가 있으면 동일한지 확인(다르면 스킵)
            if numeric_id and pid and str(pid) != str(numeric_id):
                continue
            # URL 템플릿 구성
            base = self.cfg.base.rstrip("/")
            # 템플릿에 {cmt}가 있으면 총 댓글수를 사용
            fmt_dict = {
                "base": base,
                "board": board,
                "id": (pid or numeric_id or ""),
                "page": pn,
                "cmt": total_comments or "",
            }
            url = url_template.format(**fmt_dict)
            if not url.startswith("http"):
                url = self.normalize_url(url)
            out[pn] = url
        return out

    def _collect_more_comments(self, post_url: str, pid: Optional[str], first_soup: BeautifulSoup, total_comments: Optional[int] = None) -> List[Dict]:
        numeric_id = self._post_numeric_id(pid, post_url)
        discovered = self._scan_comment_page_links(first_soup, numeric_id, total_comments=total_comments)
        if not discovered:
            return []
        # BFS over discovered pages up to a safe cap
        MAX_PAGES = 10
        visited: set[int] = set([1])
        to_visit = sorted(discovered.keys())
        all_comments: List[Dict] = []
        while to_visit and len(visited) < (MAX_PAGES + 1):
            pn = to_visit.pop(0)
            if pn in visited:
                continue
            visited.add(pn)
            url = discovered.get(pn)
            if not url:
                continue
            try:
                html = self.fetch_html(url)
                sp = soupify(html)
                # accumulate comments from this page
                cmts = extract_comments(sp, self.cfg.selectors)
                if cmts:
                    all_comments.extend(cmts)
                # discover more from this page
                more = self._scan_comment_page_links(sp, numeric_id)
                for k, v in more.items():
                    if k not in discovered and k not in visited:
                        discovered[k] = v
                        to_visit.append(k)
                to_visit = sorted(set(to_visit))
            except Exception:
                continue
        return all_comments
