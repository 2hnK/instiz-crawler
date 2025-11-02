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


LIST_HTML_WITH_HOT = """
<html>
  <body>
    <div id="ingreen">
      <table id="green_mainboard">
        <tr><td class="listsubject"><a href="/name/999?category=1&green=1"><div class="sbj">HOT 게시물<span class="cmt2">15</span></div></a></td></tr>
      </table>
    </div>
    <table id="mboard" class="mboard">
      <tr><td class="greentop no_mouseover"><a href="javascript:gt(1,'name','2','');">toggle</a></td></tr>
      <tr><td class="no_mouseover"><div id="sense14"></div></td></tr>
      <list>
        <tr id="list1"><td class="listsubject">
          <a href="/name/1001?category=1"><div class="sbj">첫번째 게시물<span class="cmt2">3</span></div><div class="list_subtitle">01.01 00:00</div></a>
        </td></tr>
        <tr id="list-ad"><td class="no_mouseover">
          <div class="content between_house">
            <div class="texthead bttitle">나에게 쓰는 편지 <span class="button2 button4"><a href="javascript:selectmenu2('x','name_self','2','');">추가하기</a></span></div>
            <div class="righttitle"><a href="https://www.instiz.net/name_self?category=2">더보기</a></div>
          </div>
        </td></tr>
        <tr id="list2"><td class="listsubject">
          <a href="/name/1002?category=1"><div class="sbj">두번째 게시물<span class="cmt2">0</span></div><div class="list_subtitle">01.01 00:01</div></a>
        </td></tr>
        <tr id="list3"><td class="listsubject">
          <a href="/name/1003?category=1"><div class="sbj">세번째 게시물<span class="cmt2">1</span></div><div class="list_subtitle">01.01 00:02</div></a>
        </td></tr>
      </list>
    </table>
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


def test_extract_list_items_skips_hot_sections():
    soup = BeautifulSoup(LIST_HTML_WITH_HOT, "lxml")
    selectors = {
        "list_scope": "#mboard list",
        "list_item_link": "td.listsubject > a",
        "list_comment_count": ".sbj span[class*='cmt']",
        "exclude_scopes": [
            "#ingreen",
            ".realchart_item",
            "#mboard td.no_mouseover",
            ".listsubject .list_subtitle",
            ".between_house",
            "#sense14",
        ],
    }
    items = extract_list_items(soup, selectors=selectors, board_path="name")
    assert [it["url"] for it in items] == [
        "/name/1001?category=1",
        "/name/1002?category=1",
        "/name/1003?category=1",
    ]
    assert items[0]["title"] == "첫번째 게시물"
    assert items[0]["comments_count"] == 3

