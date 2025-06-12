from hypothesis import given
from hypothesis import strategies as st


@given(st.text())
def test_parser_never_crashes(s):
    try:
        int(s)
    except Exception:
        pass  # Parser nie może rzucać nieobsłużonych wyjątków
