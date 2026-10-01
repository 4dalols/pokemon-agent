import os

os.environ.setdefault("PTCG_SEARCH_BUDGET", "0.1")
# tests/test_policy.py and tests/test_search.py pin the heuristic agent; the learned
# policy is covered by tests/test_imitation.py.
os.environ.setdefault("PTCG_HEURISTIC", "1")
