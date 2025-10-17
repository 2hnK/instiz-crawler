from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup


log = logging.getLogger(__name__)


def soupify(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def max_page_from_pagination(soup: BeautifulSoup, selectors: Optional[Dict] = None) -> Optional[int]:
    """페이지네이션에서 최대 page 번호를 추정합니다.

    - selectors.pagination_links: 페이지네이션 링크들을 직접 가리키는 CSS (리스트 가능)
    - selectors.pagination_scope: 페이지네이션 컨테이너를 한정하는 CSS. 이 범위 안의 a[href]만 스캔
    - 없으면 문서 전체 a[href]를 스캔(기존 동작)
    """
    max_page = None
    links: List = []
    if selectors:
        # 우선 맞춤 링크 선택자 사용
        pl = selectors.get("pagination_links")
        if pl:
            css_list = [pl] if isinstance(pl, str) else list(pl)
            for css in css_list:
                links.extend(soup.select(css))
        # 링크가 아직 없고 scope가 있으면 그 범위에서 추출
        if not links and selectors.get("pagination_scope"):
            scope = soup.select_one(selectors.get("pagination_scope"))
            if scope:
                links = scope.find_all("a", href=True)
    # 기본: 문서 전체
    if not links:
        links = soup.find_all("a", href=True)

    for a in links:
        href = a.get("href") or ""
        m = re.search(r"[?&]page=(\d+)", href)
        if m:
            try:
                val = int(m.group(1))
                if max_page is None or val > max_page:
                    max_page = val
            except ValueError:
                continue
    return max_page


def extract_list_items(
    soup: BeautifulSoup,
    selectors: Optional[Dict] = None,
    board_path: str = "name",
) -> List[Dict]:
    items: List[Dict] = []
    tried = False
    if selectors and "list_item_link" in selectors:
        tried = True
        sel = selectors["list_item_link"]
        # 리스트 범위 한정: list_scope 제공 시 그 안에서만 탐색
        scopes: List[BeautifulSoup] = []
        if selectors.get("list_scope"):
            sc = soup.select(selectors["list_scope"])
            scopes = [node for node in sc]
        if not scopes:
            scopes = [soup]

        link_css = [sel] if isinstance(sel, str) else list(sel)
        for root in scopes:
            # 제외 범위 제거(exclude_scopes)
            if selectors.get("exclude_scopes"):
                for ex in (selectors.get("exclude_scopes") or []):
                    for bad in root.select(ex):
                        try:
                            bad.extract()
                        except Exception:
                            pass
            for css in link_css:
                for a in root.select(css):
                    href = a.get("href")
                    if not href:
                        continue
                    url = href
                    title = a.get_text(strip=True)
                    if title:
                        items.append({"url": url, "title": title})
        if items:
            return dedupe_items(items)

    # Fallback: anchors pointing to posts within the board
    # - query param style: /{board}?no=123, id=, article=, artno=
    # - path style:       /{board}/123456
    pattern_q = re.compile(rf'/{re.escape(board_path)}[^\s"\']*(?:[?&](?:no|id|article|artno)=\d+)', re.I)
    pattern_p = re.compile(rf'/{re.escape(board_path)}/(\d+)(?:[^\d]|$)')
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern_q.search(href) or pattern_p.search(href):
            title = a.get_text(strip=True)
            if title:
                items.append({"url": href, "title": title})

    return dedupe_items(items)


def dedupe_items(items: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for it in items:
        key = (it.get("url"), it.get("title"))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def extract_post(
    soup: BeautifulSoup,
    selectors: Optional[Dict] = None,
) -> Dict:
    title = extract_first_text(
        soup,
        selectors.get("post_title") if selectors else None,
        fallbacks=[
            "meta[property='og:title']",
            "h3",
            "h2",
            "h1",
            "title",
        ],
    )
    # 본문 추출: post_body_strict가 있으면 우선 사용(텍스트 블록 위주)
    body: Optional[str] = None
    strict_used = False
    if selectors and selectors.get("post_body_strict"):
        strict_used = True
        body_strict = extract_first_text(
            soup,
            selectors.get("post_body_strict"),
            fallbacks=None,
            join_with="\n\n",
            excludes=(selectors.get("post_body_exclude") if selectors else None),
        )
        # strict 결과가 존재하고 의미 있으면 그대로 사용
        if body_strict and is_meaningful_text(body_strict):
            body = body_strict
        else:
            # strict 결과가 비어 있거나 무의미하면 본문은 null로 강제
            body = None
    # strict를 사용하지 않았거나, 엄격 모드가 없을 때만 fallback 고려
    if body is None and not strict_used:
        body = extract_first_text(
            soup,
            selectors.get("post_body") if selectors else None,
            fallbacks=[
                "div.memo_content",
                "div#memo_content",
                "div.board_body",
                "div.content",
                "article",
            ],
            join_with="\n\n",
            excludes=(selectors.get("post_body_exclude") if selectors else None),
        )
    if body:
        body = clean_post_text(body)
        if not is_meaningful_text(body):
            body = None
    # 이미지 검출은 별도 사용(현재 본문 null 판단에는 영향 없음)
    _ = extract_images(soup, selectors)
    # Post created time
    created_at = extract_first_text(
        soup,
        (selectors.get("post_created_at") if selectors else None),
        fallbacks=[
            "meta[property='article:published_time']",
            "meta[name='date']",
            "time[datetime]",
            "span.date",
            "em.date",
            "div.date",
        ],
    )
    # Likes and comments_count (heuristic fallbacks)
    likes = extract_first_number(
        soup,
        (selectors.get("post_likes") if selectors else None),
        fallbacks=[
            "span.like", "a.like", "em.like", "button.like",
            "span[class*='like']", "em[class*='like']", "div[class*='like']",
            "span[class*='good']", "span[class*='vote']",
        ],
    )
    comments = extract_comments(soup, selectors)
    comments_count = extract_first_number(
        soup,
        (selectors.get("post_comments_count") if selectors else None),
        fallbacks=[
            "span.comments", "a.comments", "em.comments",
            "span[class*='cmt']", "span[class*='repl']", "span[class*='comment']",
        ],
    )
    if comments_count is None:
        try:
            comments_count = len(comments)
        except Exception:
            comments_count = None
    return {
        "title": title or "",
        "body": body,  # None이면 CSV에서 빈 칸으로 출력됨
        "created_at": (created_at or "").strip(),
        "likes": likes if isinstance(likes, int) else (int(likes) if str(likes).isdigit() else None),
        "comments": comments,
        "comments_count": comments_count,
    }


def extract_first_text(
    soup: BeautifulSoup,
    selectors: Optional[Iterable[str]] = None,
    fallbacks: Optional[Iterable[str]] = None,
    join_with: str = " ",
    excludes: Optional[Iterable[str]] = None,
) -> Optional[str]:
    sel_list: List[str] = []
    if selectors:
        if isinstance(selectors, str):
            sel_list = [selectors]
        else:
            sel_list = list(selectors)
    if fallbacks:
        sel_list += list(fallbacks)

    for css in sel_list:
        nodes = soup.select(css)
        if not nodes:
            continue
        # Prefer visible text; if empty and node is <meta>, use its content/value
        collected: List[str] = []
        for n in nodes:
            # 제외 노드가 지정되면 해당 자손을 제거
            if excludes:
                try:
                    ex_list = [excludes] if isinstance(excludes, str) else list(excludes)
                    for ex in ex_list:
                        for bad in n.select(ex):
                            try:
                                bad.extract()
                            except Exception:
                                pass
                except Exception:
                    pass
            txt = n.get_text(" ", strip=True)
            if not txt:
                # Handle <meta property="og:title" content="...">
                try:
                    txt = n.get("content") or n.get("value")
                except Exception:
                    txt = None
            if txt:
                collected.append(txt)
        if collected:
            return join_with.join(collected)
    return None


def extract_comments(soup: BeautifulSoup, selectors: Optional[Dict] = None) -> List[Dict]:
    out: List[Dict] = []
    selc = None
    if selectors and "comments" in selectors:
        selc = selectors["comments"]
    if selc and isinstance(selc, dict):
        item_sel = selc.get("item")
        author_sel = selc.get("author")
        content_sel = selc.get("content")
        time_sel = selc.get("time")
        if item_sel:
            for item in soup.select(item_sel):
                author = first_text(item.select(author_sel)) if author_sel else None
                # content에서 시간/제어 텍스트 제거
                raw_content = first_text(item.select(content_sel)) if content_sel else first_text([item])
                # time 노드가 별도로 지정되어 있다면 해당 텍스트를 제거
                ctime = first_text(item.select(time_sel)) if time_sel else None
                content = clean_comment_text(raw_content, time_hint=ctime)
                if content:
                    out.append({
                        "author": author or "",
                        "content": content or "",
                        "created_at": ctime or "",
                    })
    if out:
        return out

    # Fallback heuristics: prefer leaf-level comment/reply nodes (avoid containers)
    # comments_strict_only가 true이면 휴리스틱을 사용하지 않음
    if selectors and selectors.get("comments_strict_only"):
        return out
    sel_commentish = ", ".join([
        "div[class*='comment']",
        "li[class*='comment']",
        # 제외: div[id*='comment'] (리스트 컨테이너 매칭 방지)
        "div[class*='reply']",
        "li[class*='reply']",
        "div[class*='reple']",
        "li[class*='reple']",
    ])
    commentish = soup.select(sel_commentish)
    for node in commentish:
        # 컨테이너(내부에 동일 패턴 자식이 또 있는 경우)는 건너뜀
        if node.select(sel_commentish):
            continue
        raw = node.get_text(" ", strip=True)
        cleaned = clean_comment_text(raw)
        if cleaned and len(cleaned) > 1:
            out.append({
                "author": "",
                "content": cleaned,
                "created_at": "",
            })
    return out


def first_text(nodes) -> Optional[str]:
    for n in nodes:
        txt = n.get_text(" ", strip=True)
        if txt:
            return txt
    return None


def clean_comment_text(text: Optional[str], time_hint: Optional[str] = None) -> Optional[str]:
    """댓글 본문에서 불필요한 UI 텍스트/상대시간/제어 단어를 제거합니다."""
    if not text:
        return text
    t = str(text)
    # 시간 힌트 제거
    if time_hint:
        t = t.replace(time_hint, " ")
    # 상대시간 패턴 제거: 10분 전, 3시간 전, 2일 전, 5개월 전, 1년 전 등(띄어쓰기/붙여쓰기 허용)
    t = re.sub(r"\b\d+\s*(초|분|시간|일|주|개월|년)\s*전\b", " ", t)
    t = re.sub(r"\b\d+(초|분|시간|일|주|개월|년)전\b", " ", t)
    # UI 텍스트 제거(긴 토큰 우선)
    ui_pattern = re.compile(r"(답답글|답글|스크랩|신고)")
    t = ui_pattern.sub(" ", t)
    # 다중 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


def clean_post_text(text: Optional[str]) -> Optional[str]:
    """게시글 본문에서 UI/네비게이션 잔여 텍스트를 제거합니다."""
    if not text:
        return text
    t = str(text)
    # 댓글 관련 상대시간이 본문에 섞여 들어온 경우 제거
    t = re.sub(r"\b\d+\s*(초|분|시간|일|주|개월|년)\s*전\b", " ", t)
    t = re.sub(r"\b\d+(초|분|시간|일|주|개월|년)전\b", " ", t)
    # UI/네비 텍스트 제거(긴 토큰부터)
    ui_patterns = [
        r"답답글", r"답글", r"스크랩", r"신고", r"이전글", r"다음글", r"전체목록", r"목록",
        r"원본\s*보기", r"이미지\s*원본\s*보기", r"클릭시\s*확대", r"첨부\s*이미지",
        r"링크\s*복사", r"원문", r"바로가기",
    ]
    for pat in ui_patterns:
        t = re.sub(pat, " ", t)
    # 과도한 공백 정리(줄바꿈은 유지)
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in t.splitlines()]
    t = "\n".join([ln for ln in lines if ln])
    return t or None


def is_meaningful_text(text: Optional[str]) -> bool:
    if not text:
        return False
    core = re.sub(r"[^\w가-힣]+", "", str(text))
    return len(core) >= 2


def extract_images(soup: BeautifulSoup, selectors: Optional[Dict] = None) -> List[str]:
    imgs: List[str] = []
    img_sel = None
    if selectors and isinstance(selectors, dict):
        img_sel = selectors.get("post_images")
    candidates: List[str] = []
    if img_sel:
        sel_list = [img_sel] if isinstance(img_sel, str) else list(img_sel)
        for css in sel_list:
            for im in soup.select(css):
                src = im.get("src")
                if src:
                    imgs.append(src)
        if imgs:
            return imgs
    for css in [
        "div.memo_content img",
        "div#memo_content img",
        "article img",
        "div.board_body img",
        "div.content img",
    ]:
        for im in soup.select(css):
            src = im.get("src")
            if src:
                candidates.append(src)
    seen = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            imgs.append(u)
    return imgs


def extract_first_number(
    soup: BeautifulSoup,
    selectors: Optional[Iterable[str]] = None,
    fallbacks: Optional[Iterable[str]] = None,
) -> Optional[int]:
    """Extract the first integer appearing in nodes selected by selectors or fallbacks."""
    sel_list: List[str] = []
    if selectors:
        if isinstance(selectors, str):
            sel_list = [selectors]
        else:
            sel_list = list(selectors)
    if fallbacks:
        sel_list += list(fallbacks)
    for css in sel_list:
        try:
            nodes = soup.select(css)
        except Exception:
            nodes = []
        if not nodes:
            continue
        for n in nodes:
            text = n.get_text(" ", strip=True)
            m = re.search(r"(\d{1,7})", text.replace(",", ""))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    continue
    return None
