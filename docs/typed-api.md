# Typed Question & Answer System

Laya v0.3.6 introduces a **fully type-safe API** with TypedDict question schemas, dataclass-based answer objects, and full MyPy/IDE support. This makes building decision pipelines more reliable, discoverable, and maintainable.

## Why Typed Questions?

| Benefit | Description |
|---------|-------------|
| **IDE Autocomplete** | Full IntelliSense on question construction (`ChoiceQuestion(`, `ScoreQuestion(`, `NoulQuestion(`) and answer access (`result.answers["dept"].choice`) |
| **MyPy Static Checking** | Catches type mismatches before runtime — strict mode compatible |
| **Dual Access Pattern** | Both dot access (`result.answers["dept"].choice`) AND dict access (`result["answers"]["dept"]["choice"]`) work — backward compatible |
| **Richer Criteria** | `criteria` as `dict[label, description]` gives the model more context than a bare list |
| **PEP 561 Compliant** | `py.typed` marker enables typed package distribution — your IDE and MyPy see the types automatically |

---

## Quick Start

```python
from laya import Router
from laya.typing import ChoiceQuestion, ScoreQuestion, NoulQuestion, Questions

# Preload checkpoints for instant sub-35ms routing
router = Router(preload=True)

# State in any language or schema (text, JSON dict, email, or conversation)
state = {
    "from": "user@acme.com",
    "subject": "Duplicate charge on invoice #4411 — need refund ASAP",
    "body": "Hi, we were billed twice for March. Please refund the duplicate today "
            "or we will cancel our plan. This is the second time this has happened."
}

# Define typed questions with full IDE autocomplete & type safety
questions: Questions = {
    "department": ChoiceQuestion(
        type="choice",
        instructions="Which department should handle this request?",
        criteria={
            "billing": "invoices, payments, refunds, charge disputes",
            "technical": "bugs, outages, system errors, API issues",
            "sales": "pricing, new contracts, upgrades",
            "account": "login, password, subscription, profile changes",
            "general": "everything else"
        }
    ),
    "urgency": ScoreQuestion(
        type="score",
        instructions="How urgent is this request on a 4-level scale?",
        criteria=["not urgent", "soon", "high", "critical deadline or blocking issue"]
    ),
    "churn_risk": NoulQuestion(
        type="noul",
        instructions="Does the user threaten to cancel, leave, or express strong dissatisfaction?",
    ),
    "refund_requested": NoulQuestion(
        type="noul",
        instructions="Does the user explicitly request a refund?",
    ),
}

# English state -> automatically routed to laya (ModernBERT-large, ~39 ms)
res_en = router.predict(state, questions)
print("Department :", res_en.answers["department"].choice)  # -> billing (confidence: 0.94)
print("Routing    :", res_en.routing.model)                 # -> english

# Hindi state -> automatically routed to laya-multilingual (mmBERT-base, ~33 ms)
res_hi = router.predict({"body": "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}, questions)
print("Department :", res_hi.answers["department"].choice)  # -> billing (confidence: 0.86)
print("Routing    :", res_hi.routing.model)                 # -> multilingual

# Explicit override when you want a specific checkpoint
res_td = router.predict(state, questions, model="typed-decisions")
```

---

## Question Types

### ChoiceQuestion — Pick One Option

```python
from laya.typing import ChoiceQuestion

# With rich descriptions (recommended — gives model more context)
q = ChoiceQuestion(
    type="choice",
    instructions="Which department should handle this request?",
    criteria={
        "billing": "invoices, payments, refunds, charge disputes",
        "technical": "bugs, outages, system errors, API issues",
        "sales": "pricing, new contracts, upgrades",
        "general": "everything else"
    }
)

# Or just labels (backward compatible)
q = ChoiceQuestion(
    type="choice",
    instructions="Which department?",
    criteria=["billing", "technical", "sales", "general"]
)
```

| Field | Type | Required |
|-------|------|----------|
| `type` | `Literal["choice"]` | ✅ |
| `instructions` | `str` | ✅ |
| `criteria` | `dict[str, str] \| list[str]` | ✅ |

**Recommendation**: Use `dict[str, str]` for `criteria` — the descriptions help the model disambiguate similar options.

---

### ScoreQuestion — Rate on Ordinal Scale

```python
from laya.typing import ScoreQuestion

q = ScoreQuestion(
    type="score",
    instructions="How urgent is this request on a 4-level scale?",
    criteria=["not urgent", "soon", "high", "critical deadline or blocking issue"]
)
```

| Field | Type | Required |
|-------|------|----------|
| `type` | `Literal["score"]` | ✅ |
| `instructions` | `str` | ✅ |
| `criteria` | `list[str]` | ✅ |

The list index is the level (0 = lowest). The model returns an expected value (e.g., `2.8` on 0–3 scale).

---

### NoulQuestion — Binary Probability P(true)

```python
from laya.typing import NoulQuestion

# With optional true/false descriptions
q = NoulQuestion(
    type="noul",
    instructions="Does the user threaten to cancel or leave?",
    criteria={
        "true": "explicit threat to cancel, close account, or switch providers",
        "false": "no threat, or user is satisfied/neutral"
    }
)

# Or minimal (criteria is optional for noul)
q = NoulQuestion(
    type="noul",
    instructions="Is this spam?"
)
```

| Field | Type | Required |
|-------|------|----------|
| `type` | `Literal["noul"]` | ✅ |
| `instructions` | `str` | ✅ |
| `criteria` | `NotRequired[dict[str, str]]` | ❌ |

Returns a calibrated probability `P(true) ∈ [0, 1]`.

---

## Questions Type

```python
from laya.typing import Questions, ChoiceQuestion, ScoreQuestion, NoulQuestion

questions: Questions = {
    "department": ChoiceQuestion(...),
    "urgency": ScoreQuestion(...),
    "churn_risk": NoulQuestion(...),
}
```

`Questions` is a type alias for `dict[str, Question]` where `Question = ChoiceQuestion | ScoreQuestion | NoulQuestion`.

---

## Typed Answer Objects (PredictResult)

Every prediction returns a `PredictResult` with **both dot and dict access**:

```python
from laya import Router
from laya.agent import PredictResult

state = {"body": "I was charged twice, refund me now or I'm leaving!"}
router = Router(preload=True)

result: PredictResult = router.predict(state, questions)

# Dot access (IDE autocomplete works!)
print(result.answers["department"].choice)        # "billing"
print(result.answers["department"].probabilities) # {"billing": 0.94, "technical": 0.03, ...}
print(result.answers["department"].confidence)    # 0.94 (normalized entropy)
print(result.answers["department"].answer_confidence) # 0.99 (calibrated)
print(result.answers["department"].action)        # {"act_probability": 0.99}

print(result.answers["urgency"].score)            # 2.8 (expected value on 0-3 scale)
print(result.answers["urgency"].legend)           # {"0": "not urgent", "1": "soon", "2": "high", "3": "critical..."}
print(result.answers["urgency"].probabilities)    # {"0": 0.02, "1": 0.08, "2": 0.25, "3": 0.65}
print(result.answers["urgency"].confidence)       # 0.81

print(result.answers["churn_risk"].noul)          # 0.87 (P(true) = 87% churn risk)
print(result.answers["churn_risk"].confidence)    # 0.87 (max(p, 1-p))
print(result.answers["churn_risk"].answer_confidence) # 0.87 (calibrated)

print(result.usage)                               # {"input_tokens": 67, "output_tokens": 0}
print(result.routing.model)                       # "english" (or "multilingual" for non-Latin scripts)

# Dict access also works (backward compatible)
print(result["answers"]["department"]["choice"])  # "billing"
print(result["routing"]["model"])                 # "english"
```

---

## Answer Type Reference

### ChoiceAnswer

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["choice"]` | Always `"choice"` |
| `choice` | `str` | The selected label |
| `probabilities` | `dict[str, float]` | Probability per option (sums to 1.0) |
| `confidence` | `float` | Normalized entropy confidence (0–1) |
| `answer_confidence` | `float` | Calibrated confidence from proper scoring rule |
| `action` | `dict[str, float]` | `{"act_probability": float}` — P(act on this) |

### ScoreAnswer

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["score"]` | Always `"score"` |
| `score` | `float` | Expected value (e.g., `2.8` on 0–3) |
| `legend` | `dict[str, str]` | Level index → description mapping |
| `probabilities` | `dict[str, float]` | Probability per level index |
| `confidence` | `float` | Normalized entropy confidence |
| `answer_confidence` | `float` | Calibrated confidence |
| `action` | `dict[str, float]` | `{"act_probability": float}` |

### NoulAnswer

| Field | Type | Description |
|-------|------|-------------|
| `type` | `Literal["noul"]` | Always `"noul"` |
| `noul` | `float` | Calibrated P(true) ∈ [0, 1] |
| `confidence` | `float` | `max(p, 1-p)` — certainty about the binary outcome |
| `answer_confidence` | `float` | Calibrated confidence (identical to `confidence` for binary) |
| `action` | `dict[str, float]` | `{"act_probability": float}` |

### PredictResult

| Field | Type | Description |
|-------|------|-------------|
| `model` | `str` | Checkpoint name used |
| `answers` | `dict[str, Answer]` | Question ID → typed answer |
| `usage` | `UsageDict` | `{"input_tokens": int, "output_tokens": int}` |
| `routing` | `RouteDecisionDict \| None` | Router decision metadata (when using Router) |

---

## Using with Agent Directly

```python
import laya
from laya.typing import ChoiceQuestion, ScoreQuestion, NoulQuestion, Questions

# Load a specific checkpoint directly from the hub
agent = laya.load("convaiinnovations/laya")                           # English root
agent_ml = laya.load("convaiinnovations/laya", subfolder="multilingual") # 100+ languages
agent_td = laya.load("convaiinnovations/laya", subfolder="typed-decisions")

# Define typed questions (with full IDE autocomplete)
questions: Questions = {
    "department": ChoiceQuestion(
        type="choice",
        instructions="Which department should handle this?",
        criteria={
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, system errors",
            "sales": "pricing, new contracts",
            "general": "everything else"
        }
    ),
    "urgency": ScoreQuestion(
        type="score",
        instructions="How urgent is this request?",
        criteria=["not urgent", "soon", "critical deadline or blocking issue"]
    ),
    "churn_risk": NoulQuestion(
        type="noul",
        instructions="Does the user threaten to cancel or leave?"
    ),
}

# Run all questions in ONE single forward pass (~35 ms on GPU)
result = agent.predict(state, questions)

# Dot access (IDE autocomplete)
print("Department :", result.answers["department"].choice)   # -> billing (confidence: 0.94)
print("Urgency    :", result.answers["urgency"].score)        # -> 1.84 / 2.0
print("Churn Risk :", result.answers["churn_risk"].noul)       # -> 0.892 (89.2% probability)

# Dict access also works (backward compatible)
print("Department :", result["answers"]["department"]["choice"])
```

---

## FastAPI Integration with Pydantic

```python
from fastapi import FastAPI
from pydantic import BaseModel
from laya import Router
from laya.typing import (
    ChoiceQuestion, ScoreQuestion, NoulQuestion,
    Questions
)
from laya.agent import PredictResult

app = FastAPI()
router = Router(preload=True)

# Define your schema with Pydantic for request/response validation
class TicketRequest(BaseModel):
    from_email: str
    subject: str
    body: str

class DepartmentChoice(BaseModel):
    choice: str
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: dict[str, float]

class UrgencyScore(BaseModel):
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float
    answer_confidence: float
    action: dict[str, float]

class ChurnRisk(BaseModel):
    noul: float
    confidence: float
    answer_confidence: float
    action: dict[str, float]

class TicketResponse(BaseModel):
    model: str
    department: DepartmentChoice
    urgency: UrgencyScore
    churn_risk: ChurnRisk
    usage: dict[str, int]
    routing: dict[str, str]

# Questions defined once at startup
QUESTIONS: Questions = {
    "department": ChoiceQuestion(
        type="choice",
        instructions="Route this support ticket to the correct department",
        criteria={
            "billing": "invoices, payments, refunds, charge disputes",
            "technical": "bugs, outages, system errors, API issues",
            "account": "login, password, subscription, profile changes",
            "general": "everything else"
        }
    ),
    "urgency": ScoreQuestion(
        type="score",
        instructions="Rate the urgency of this issue on a 4-level scale",
        criteria=["low", "medium", "high", "critical"]
    ),
    "is_spam": NoulQuestion(
        type="noul",
        instructions="Is this message spam or unsolicited commercial content?",
        criteria={"true": "clear spam, promotional, or bot-generated", "false": "legitimate user message"}
    ),
    "churn_risk": NoulQuestion(
        type="noul",
        instructions="Does the user threaten to cancel, leave, or express strong dissatisfaction?",
    ),
}

@app.post("/triage", response_model=TicketResponse)
async def triage_ticket(request: TicketRequest):
    state = {
        "from": request.from_email,
        "subject": request.subject,
        "body": request.body
    }
    result: PredictResult = router.predict(state, QUESTIONS)
    return result  # FastAPI serializes the _DictCompatible dataclasses automatically
```

---

## MyPy Configuration

The package includes a strict MyPy config in `pyproject.toml`. To enable checking in your project:

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = false
namespace_packages = true

# Third-party libs without stubs (already in laya's config)
[[tool.mypy.overrides]]
module = "torch.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "transformers.*"
ignore_missing_imports = true

# ... etc for safetensors, huggingface_hub, tokenizers, numpy, uvicorn, starlette
```

Run MyPy on your code:
```bash
mypy your_code.py
```

---

## Backward Compatibility

**All existing dict-based code continues to work unchanged:**

```python
# This still works exactly as before
questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department?",
        "criteria": ["billing", "technical", "sales"]
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["low", "medium", "high"]
    }
}

result = router.predict(state, questions)
print(result["answers"]["department"]["choice"])  # dict access
```

The new typed API is **additive** — you can migrate incrementally.

---

## Migration Guide

| Before (dict) | After (typed) |
|--------------|---------------|
| `{"type": "choice", "instructions": "...", "criteria": ["a", "b"]}` | `ChoiceQuestion(type="choice", instructions="...", criteria={"a": "...", "b": "..."})` |
| `{"type": "score", "instructions": "...", "criteria": ["low", "high"]}` | `ScoreQuestion(type="score", instructions="...", criteria=["low", "high"])` |
| `{"type": "noul", "instructions": "..."}` | `NoulQuestion(type="noul", instructions="...")` |
| `result["answers"]["q"]["choice"]` | `result.answers["q"].choice` (or keep dict access) |
| `result["answers"]["q"]["score"]` | `result.answers["q"].score` |
| `result["answers"]["q"]["noul"]` | `result.answers["q"].noul` |

---

## API Reference

::: laya.typing.ChoiceQuestion
::: laya.typing.ScoreQuestion
::: laya.typing.NoulQuestion
::: laya.typing.Questions
::: laya.agent.PredictResult
::: laya.agent.ChoiceAnswer
::: laya.agent.ScoreAnswer
::: laya.agent.NoulAnswer
::: laya.typing.UsageDict
::: laya.typing.RouteDecisionDict
::: laya.typing.State