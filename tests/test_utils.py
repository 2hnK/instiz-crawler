import datetime as _dt

from instiz_crawler.utils import parse_date, iter_months, to_month_str


def test_parse_date_and_months():
    s = parse_date("2024-01-01")
    e = parse_date("2024-03-15")
    months = list(iter_months(s, e, step=1))
    assert months[0][0] == _dt.date(2024, 1, 1)
    assert months[0][1] == _dt.date(2024, 2, 1)
    assert months[1][0] == _dt.date(2024, 2, 1)
    assert months[1][1] == _dt.date(2024, 3, 1)
    # inclusive end: 2024-03-15 includes 3월 스타트
    assert months[2][0] == _dt.date(2024, 3, 1)
    assert months[2][1] == _dt.date(2024, 4, 1)
    assert to_month_str(months[0][0]) == "2024-01"

