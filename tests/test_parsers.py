from bs4 import BeautifulSoup

from instiz_crawler.parsers import max_page_from_pagination, extract_list_items, extract_post


LIST_HTML = """
<html>
  <body>
    <a href="/name?page=1&category=1">1</a>
    <a href="/name?page=3&category=1">3</a>
    <table>
      <tr><td class="t_left"><a href="/name?no=111">제목 A</a></td></tr>
      <tr><td class="t_left"><a href="/name?no=222">제목 B</a></td></tr>
    </table>
  </body>
 </html>
"""


POST_HTML = """
<html>
  <head><meta property="og:title" content="제목 A"/></head>
  <body>
    <div class="memo_content">본문 내용 A</div>
    <ul>
      <li class="comment"><span class="author">닉1</span><span class="text">댓글1</span><span class="time">09:00</span></li>
      <li class="comment"><span class="author">닉2</span><span class="text">댓글2</span><span class="time">10:00</span></li>
    </ul>
  </body>
 </html>
"""


def test_max_page_from_pagination():
    soup = BeautifulSoup(LIST_HTML, "lxml")
    assert max_page_from_pagination(soup) == 3


def test_extract_list_items():
    soup = BeautifulSoup(LIST_HTML, "lxml")
    items = extract_list_items(soup, selectors={"list_item_link": ["td.t_left > a"]}, board_path="name")
    assert len(items) == 2
    assert items[0]["url"].startswith("/name?no=")
    assert items[0]["title"]


def test_extract_post_title_body_comments():
    soup = BeautifulSoup(POST_HTML, "lxml")
    post = extract_post(soup, selectors=None)
    assert post["title"] == "제목 A"
    assert "본문 내용" in post["body"]
    assert len(post["comments"]) == 2

