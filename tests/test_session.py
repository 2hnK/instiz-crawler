from instiz_crawler.session import detect_login_required


def test_detect_login_required_by_keywords():
    html = """
    <html><body>
      <form action="/user/login">
        <label>아이디</label>
        <input type="text"/>
        <label>비밀번호</label>
        <input type="password"/>
      </form>
    </body></html>
    """
    assert detect_login_required(html, url="https://www.instiz.net/user/login") is True


def test_detect_login_required_false_for_normal_page():
    html = "<html><body><h1>일반 페이지</h1></body></html>"
    assert detect_login_required(html, url="https://www.instiz.net/name?page=1") is False

