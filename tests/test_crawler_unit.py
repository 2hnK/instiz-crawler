from datetime import date

import types

from instiz_crawler.crawler import InstizCrawler, CrawlConfig


class DummySession:
    def __init__(self):
        self.proxies = {}
        self.headers = {}


def test_build_search_url_and_pages(tmp_path):
    sess = DummySession()
    cfg = CrawlConfig()
    c = InstizCrawler(sess, cfg, output_dir=str(tmp_path), verbose=False, dry_run=True)

    # URL 포맷 확인 (공백 대신 + 유지)
    url = c.build_search_url(date(2024, 1, 1), date(2024, 2, 1), page=11)
    assert "page=11" in url
    assert "starttime=2024/01/01+00:00" in url
    assert "endtime=2024/02/01+00:00" in url
    # 시간대 지정 시 마지막 날짜는 월의 말일로 설정된다.
    ranged_url = c.build_search_url(
        date(2024, 1, 1),
        date(2024, 2, 1),
        page=2,
        start_time="06:00",
        end_time="11:59",
    )
    assert "page=2" in ranged_url
    assert "starttime=2024/01/01+06:00" in ranged_url
    assert "endtime=2024/01/31+11:59" in ranged_url

    # 페이지 빌드 로직 확인
    pages = c._build_pages(start=1, total=47, step=10, pages_needed=5)
    # 1,11,21,31,41 그리고 total(47) 추가 → 1,11,21,31,41,47
    assert pages[0] == 1 and pages[-1] == 47
    assert 11 in pages and 41 in pages
