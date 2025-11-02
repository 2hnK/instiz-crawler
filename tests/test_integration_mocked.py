import csv
from datetime import date

from instiz_crawler.crawler import InstizCrawler, CrawlConfig


LIST_HTML = """
<html>
  <body>
    <a href="/name?page=1&category=1">1</a>
    <a href="/name?page=3&category=1">3</a>
    <table>
      <tr>
        <td class="t_left">
          <a href="/name?no=111">
            <div class="meta"><span class="reply">[3]</span></div>
            <span class="title">제목 A</span>
          </a>
        </td>
      </tr>
      <tr>
        <td class="t_left">
          <a href="/name?no=222">
            <div class="meta"><span class="reply">[5]</span></div>
            <span class="title">제목 B</span>
          </a>
        </td>
      </tr>
    </table>
  </body>
 </html>
"""

POST_HTML_A = """
<html>
  <head><meta property="og:title" content="제목 A"/></head>
  <body>
    <div class="memo_content">본문 내용 A</div>
    <ul>
      <li class="comment"><span class="author">닉1</span><span class="text">댓글1</span><span class="time">09:00</span></li>
    </ul>
  </body>
 </html>
"""

POST_HTML_B = """
<html>
  <head><meta property="og:title" content="제목 B"/></head>
  <body>
    <div class="memo_content">본문 내용 B</div>
    <ul>
      <li class="comment"><span class="author">닉2</span><span class="text">댓글2</span><span class="time">10:00</span></li>
      <li class="comment"><span class="author">닉3</span><span class="text">댓글3</span><span class="time">11:00</span></li>
    </ul>
  </body>
 </html>
"""


class FakeResp:
    def __init__(self, text, url):
        self.text = text
        self.url = url


class FakeSession:
    def __init__(self):
        self.headers = {}
        self.proxies = {}

    def get(self, url, timeout=15):
        # 간단 매핑: 목록 vs 상세
        if "page=" in url:
            return FakeResp(LIST_HTML, url)
        if "no=111" in url:
            return FakeResp(POST_HTML_A, url)
        if "no=222" in url:
            return FakeResp(POST_HTML_B, url)
        return FakeResp("<html></html>", url)


def test_integration_csv_outputs(tmp_path):
    sess = FakeSession()
    cfg = CrawlConfig(posts_per_page=20, target_posts_per_month=2)
    c = InstizCrawler(sess, cfg, output_dir=str(tmp_path), verbose=False, dry_run=False, output_format="csv")

    c.run_month(date(2024, 1, 1), date(2024, 2, 1))

    csv_path = tmp_path / "instiz_2024-01.csv"
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    titles = {row["title"]: row for row in rows}
    assert "제목 A" in titles and "제목 B" in titles
    # 댓글 수는 목록에서 제공된 값으로 기입된다.
    assert titles["제목 A"]["comments_count"] == "3"
    assert titles["제목 B"]["comments_count"] == "5"
    # (변경) CSV 메타 컬럼 축소: timerange 컬럼 제거됨
    assert "timerange" not in rows[0]
