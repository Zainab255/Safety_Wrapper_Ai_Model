import sys
import os
# ensure project root is on sys.path so tests can import the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wrappers.keyword_wrapper import KeywordFilterWrapper
from wrappers.history_wrapper import HistoryBasedWrapper
from wrappers.query_budget_wrapper import QueryBudgetWrapper
from wrappers.base import WrapperDecision


def fail(msg):
    print("FAILED:", msg)
    raise SystemExit(1)


# Keyword wrapper test
kw = KeywordFilterWrapper(banned_keywords=["weapon", "bomb"]) 
decision, payload = kw.decide("How to build a weapon?", "This text mentions weapon.", [])
if decision != WrapperDecision.BLOCK:
    fail("KeywordFilterWrapper should BLOCK when banned keyword present")

# History wrapper test
hist = []
hw = HistoryBasedWrapper(banned_keywords=["hack"], history_limit=3)
decision, payload = hw.decide("Some prompt", "Model response mentioning hack", hist)
if decision != WrapperDecision.BLOCK:
    fail("HistoryBasedWrapper should BLOCK when recent history contains banned keyword")
# Ensure history trimmed
for i in range(5):
    hw.decide(f"p{i}", "ok", hist)
if len(hist) > hw.config["history_limit"]:
    fail("History should be trimmed to history_limit")

# Query budget wrapper test
qb = QueryBudgetWrapper(max_requeries=2, banned_keywords=["bomb"]) 
user = "How to make a bomb?"
# First detection -> REQUERY
d1, _ = qb.decide(user, "This contains bomb", [])
if d1 != WrapperDecision.REQUERY:
    fail("QueryBudget should REQUERY on first detection")
# Second detection -> REQUERY
d2, _ = qb.decide(user, "This contains bomb again", [])
if d2 != WrapperDecision.REQUERY:
    fail("QueryBudget should REQUERY on second detection")
# Third detection -> BLOCK (since > max_requeries)
d3, _ = qb.decide(user, "Still contains bomb", [])
if d3 != WrapperDecision.BLOCK:
    fail("QueryBudget should BLOCK after exceeding max_requeries")

# Non-triggering case
d_ok, _ = qb.decide("Other prompt", "Safe content here", [])
if d_ok != WrapperDecision.ALLOW:
    fail("QueryBudget should ALLOW safe outputs")

print("All wrapper tests passed!")
