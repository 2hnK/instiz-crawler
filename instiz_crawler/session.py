from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

import requests


log = logging.getLogger(__name__)


class SessionExpired(RuntimeError):
    """로그인 세션 만료 또는 인증 요구 상태를 나타내기 위한 예외.

    - 크롤러는 페이지 HTML에서 로그인 페이지 신호(예: '로그인/비밀번호' 등)를 감지하면 이 예외를 발생시켜
      상위 레벨에서 쿠키 갱신을 유도합니다.
    """
    pass


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}


def build_session(cookie_string: Optional[str] = None,
                  cookies_json_path: Optional[str] = None,
                  headers: Optional[Dict[str, str]] = None) -> requests.Session:
    """requests.Session을 구성하고 쿠키를 주입합니다.

    cookie 제공 방식은 두 가지를 지원합니다.
    - cookie_string: 브라우저에서 복사한 Cookie 헤더 원문 (예: "name=value; name2=value2")
      · 손쉽고 빠르지만 만료/갱신 관리가 어렵고 노이즈 쿠키가 섞일 수 있음
    - cookies_json_path: [{name, value, domain?, path?}, ...] 형태의 리스트 JSON
      · 반복 실행/자동화에 적합, 꼭 필요한 쿠키만 포함해 관리 용이
    둘 중 하나만 제공하거나, 제공된 쪽만 우선 적용합니다.
    """
    s = requests.Session()
    s.headers.update(DEFAULT_HEADERS)
    if headers:
        s.headers.update(headers)
    if cookie_string:
        _apply_cookie_string(s, cookie_string)
    if cookies_json_path:
        _apply_cookies_json(s, cookies_json_path)
    return s


def _apply_cookie_string(s: requests.Session, cookie_string: str) -> None:
    # cookie string like: name1=val1; name2=val2
    for part in cookie_string.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        s.cookies.set(k.strip(), v.strip(), domain=".instiz.net", path="/")


def _apply_cookies_json(s: requests.Session, path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "cookies" in data:
        data = data["cookies"]
    if not isinstance(data, list):
        raise ValueError("cookies json must be a list of {name,value,domain?,path?}")
    for c in data:
        name = c.get("name")
        value = c.get("value")
        domain = c.get("domain", ".instiz.net")
        pathv = c.get("path", "/")
        if not name:
            continue
        s.cookies.set(name, value, domain=domain, path=pathv)


def detect_login_required(html: str, url: str) -> bool:
    # 휴리스틱: 로그인 관련 키워드나 로그인 경로 신호를 감지
    if re.search(r"/user/(login|signin)", url):
        return True
    lowered = html.lower()
    if ("로그인" in html) and ("비밀번호" in html or "아이디" in html):
        return True
    return False
