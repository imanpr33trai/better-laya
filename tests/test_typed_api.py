"""Tests for the typed question/answer API (typing module only).

Run: python tests/test_typed_api.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import typing module directly without triggering laya package init
import importlib.util
typing_spec = importlib.util.spec_from_file_location("laya.typing", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "laya", "typing.py"))
typing_module = importlib.util.module_from_spec(typing_spec)
typing_spec.loader.exec_module(typing_module)

ChoiceQuestion = typing_module.ChoiceQuestion
ScoreQuestion = typing_module.ScoreQuestion
NoulQuestion = typing_module.NoulQuestion
Questions = typing_module.Questions
UsageDict = typing_module.UsageDict
RouteDecisionDict = typing_module.RouteDecisionDict
State = typing_module.State
Question = typing_module.Question
QType = typing_module.QType

from dataclasses import dataclass, fields

PASS, FAIL = [], []


def check(name, got, want):
    if got == want:
        PASS.append(name)
    else:
        FAIL.append("%s: got %r, want %r" % (name, got, want))


def check_true(name, cond, detail=""):
    if cond:
        PASS.append(name)
    else:
        FAIL.append("%s%s" % (name, ": " + detail if detail else ""))


def check_type(name, obj, expected_type):
    if isinstance(obj, expected_type):
        PASS.append(name)
    else:
        FAIL.append("%s: expected %s, got %s" % (name, expected_type, type(obj)))


# ============================================================
# Test TypedDict question schemas
# ============================================================
def test_typed_question_schemas():
    # ChoiceQuestion with dict criteria
    cq = ChoiceQuestion(
        type="choice",
        instructions="Which department?",
        criteria={"billing": "invoices", "technical": "bugs"}
    )
    check("ChoiceQuestion/type", cq["type"], "choice")
    check("ChoiceQuestion/instructions", cq["instructions"], "Which department?")
    check("ChoiceQuestion/criteria", cq["criteria"], {"billing": "invoices", "technical": "bugs"})
    
    # ChoiceQuestion with list criteria
    cq2 = ChoiceQuestion(
        type="choice",
        instructions="Which department?",
        criteria=["billing", "technical"]
    )
    check("ChoiceQuestion/list criteria", cq2["criteria"], ["billing", "technical"])
    
    # ScoreQuestion
    sq = ScoreQuestion(
        type="score",
        instructions="Rate urgency",
        criteria=["low", "medium", "high"]
    )
    check("ScoreQuestion/type", sq["type"], "score")
    check("ScoreQuestion/criteria", sq["criteria"], ["low", "medium", "high"])
    
    # NoulQuestion
    nq = NoulQuestion(
        type="noul",
        instructions="Is this spam?",
        criteria={"true": "spam", "false": "legitimate"}
    )
    check("NoulQuestion/type", nq["type"], "noul")
    check("NoulQuestion/criteria", nq["criteria"], {"true": "spam", "false": "legitimate"})
    
    # NoulQuestion without criteria
    nq2 = NoulQuestion(
        type="noul",
        instructions="Is this spam?"
    )
    check("NoulQuestion/no criteria", "criteria" not in nq2, True)
    
    # Questions union
    questions: Questions = {
        "dept": cq,
        "urgency": sq,
        "spam": nq,
    }
    check("Questions/dict", set(questions.keys()), {"dept", "urgency", "spam"})
    
    # Type aliases
    check("Question union/type", ChoiceQuestion.__name__, "ChoiceQuestion")
    check("QType/type", QType.__name__, "Literal")


# ============================================================
# Test State type
# ============================================================
def test_state_type():
    # Just verify type alias exists
    check_true("State/alias exists", State is not None)


# ============================================================
# Test UsageDict and RouteDecisionDict
# ============================================================
def test_result_typeddicts():
    # UsageDict
    usage: UsageDict = {"input_tokens": 50, "output_tokens": 0}
    check("UsageDict", usage, {"input_tokens": 50, "output_tokens": 0})
    
    # RouteDecisionDict
    route: RouteDecisionDict = {
        "model": "english",
        "repo": "convaiinnovations/laya",
        "reason": "English Latin text",
        "detection": {"script": "latin", "is_english": True},
        "workflow": None,
    }
    check("RouteDecisionDict", route["model"], "english")
    check("RouteDecisionDict/detection", route["detection"], {"script": "latin", "is_english": True})


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("Running typed API tests (typing module only)...\n")
    
    test_typed_question_schemas()
    test_state_type()
    test_result_typeddicts()
    
    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAIL", f)
    
    if FAIL:
        sys.exit(1)
    else:
        print("all typed API tests passed")