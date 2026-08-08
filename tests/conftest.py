"""Pytest collection rules for the BRITTAIN repository."""

# test_fim.py is an interactive checkpoint evaluation program. It parses command
# line arguments and loads a large checkpoint at import time, so it is not a unit
# test module. Keep it available as a manual test without collecting it in pytest.
collect_ignore = ["test_fim.py"]
