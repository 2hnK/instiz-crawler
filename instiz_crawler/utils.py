import json
import os
import random
import re
import time
from datetime import datetime, date
from typing import Dict, Iterable, List, Optional, Tuple


ISO_DATE_FMT = "%Y-%m-%d"


def parse_date(d: str) -> date:
    return datetime.strptime(d, ISO_DATE_FMT).date()


def month_add(year: int, month: int, delta: int) -> Tuple[int, int]:
    # Adds delta months to (year, month)
    idx = (year * 12 + (month - 1)) + delta
    y = idx // 12
    m = (idx % 12) + 1
    return y, m


def iter_months(start: date, end: date, step: int = 1) -> Iterable[Tuple[date, date]]:
    """
    Yield (month_start, month_end) for [start, end] inclusive.
    month_end is the first day of the next month.
    """
    if step <= 0:
        raise ValueError("step must be >= 1")
    y, m = start.year, start.month
    while True:
        ms = date(y, m, 1)
        ny, nm = month_add(y, m, step)
        me = date(ny, nm, 1)
        if ms > end:
            break
        yield ms, me
        y, m = ny, nm


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def extract_query_int(url: str, key: str) -> Optional[int]:
    m = re.search(rf"[?&]{re.escape(key)}=(\d+)", url)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def sleep_jitter(min_s: float, max_s: float) -> None:
    if max_s < min_s:
        max_s = min_s
    t = random.uniform(min_s, max_s)
    time.sleep(t)


def load_json(path: str) -> Dict:
    """단순 JSON 로더(현재 코드 경로에서는 미사용). 필요 시 임포트하여 사용하세요."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_month_str(d: date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def parse_datetime_loose(s: Optional[str]) -> Optional[datetime]:
    """Best-effort parse of various datetime strings.

    - Tries ISO8601 (including timezone like +09:00, or Z)
    - Falls back to extracting YYYY-MM-DD (or with / or .) from the string
    """
    if not s:
        return None
    txt = s.strip()
    if not txt:
        return None
    # Handle 'Z' suffix
    zfix = txt.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(zfix)
    except Exception:
        pass
    # Extract first YYYY-MM-DD (or YYYY/MM/DD or YYYY.MM.DD)
    m = re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", txt)
    if m:
        try:
            y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, da)
        except Exception:
            return None
    return None


def format_ampm_str(s: Optional[str]) -> str:
    """문자열 형태의 datetime을 'YYYY-MM-DD hh:mm:ss AM/PM' 형식으로 변환.

    - 파싱 실패 시 원문(또는 빈 문자열)을 반환
    """
    if not s:
        return ""
    dt = parse_datetime_loose(s)
    if not dt:
        return s
    try:
        return dt.strftime("%Y-%m-%d %I:%M:%S %p")
    except Exception:
        return s
