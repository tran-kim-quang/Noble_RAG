# Noble RAG Refactor Plan
## Two-Route Architecture (`knowledge_base` + `chat`)
### Implementation Guide by Phase

## 1. Purpose

This document describes a **stable, phased implementation plan** to refactor the current `feature/sale-agent` branch into a cleaner two-route architecture:

- `knowledge_base`
- `chat`

The goal is to:

- keep query rewriting and multi-intent decomposition
- remove the current three-way routing split (`RAG`, `SEARCH`, `OTHER`) from the top-level API flow
- add a **cosine-score-based fallback to web search**
- preserve or recreate the **sales state machine**
- make `chat` the single response layer that always talks to the user
- keep internal knowledge facts and external search facts clearly separated

This plan is designed to be implemented **incrementally**, with **clear rollback points**, **small PRs**, and **testable milestones**.

---

## 2. Current Baseline

From the current branch structure:

- `service/RAG/rag/retriever.py`
  - contains `route_query()`
  - contains `plan_query_adaptive()`
  - contains `query_rag()`, `query_rag_stream()`
  - contains `summarize_search_answer()`
- `service/RAG/api/routes_sales.py`
  - currently calls `_run_sales_or_search()`
  - routes into `RAG`, `SEARCH`, `OTHER`, `MIXED`, or `SALES`
- `service/RAG/api/routes_query.py`
  - forwards `/query/stream` into the sales chat stream
- `service/RAG/sales/graph.py`
  - defines the sales graph pipeline
- `service/RAG/sales/state_machine.py`
  - contains deterministic state transitions
- `service/RAG/sales/graph_state.py`
  - defines the graph state object
- `service/RAG/models/api_models.py`
  - contains request/response contracts

### Current pain points

1. Top-level routing is split across `RAG`, `SEARCH`, `OTHER`, and sometimes falls back into sales.
2. Search results are summarized in a separate route instead of being normalized into a unified response context.
3. User-facing chat behavior is split between:
   - sales graph
   - free-form `OTHER` flow
   - direct RAG output
4. There is no single, explicit contract between:
   - internal KB evidence
   - external web evidence
   - chat orchestration
5. Routing currently depends on an LLM route decision, which adds latency and complexity.

---

## 3. Target Architecture

## 3.1 High-level design

### Route 1: `knowledge_base`

Responsibilities:

- normalize the query
- rewrite the query
- detect multi-intent
- decompose into subqueries
- retrieve internal KB candidates
- compute cosine confidence from the KB retrieval result
- decide whether KB evidence is enough
- if KB confidence is low, optionally call web search
- return a **structured evidence payload**
- never talk directly to the user

### Route 2: `chat`

Responsibilities:

- always produce the final user-facing response
- use:
  - personal system prompt
  - chat history
  - sales state machine
  - structured evidence from `knowledge_base` when available
- ask follow-up questions when user profile is incomplete
- recommend, compare, or explain products when sufficient evidence exists
- keep the conversation warm, natural, and sales-oriented

---

## 3.2 Core principle

**`knowledge_base` retrieves and packages evidence. `chat` speaks.**

That separation is the most important design rule in this refactor.

---

## 4. Refactor Strategy

Implement in the following phases:

- **Phase 0**: Stabilize the current code before changing behavior
- **Phase 1**: Extract a reusable planner from `plan_query_adaptive()`
- **Phase 2**: Build cosine-based KB confidence evaluation
- **Phase 3**: Build structured payload between `knowledge_base` and `chat`
- **Phase 4**: Make `chat` the only user-facing response route
- **Phase 5**: Preserve and adapt the sales state machine
- **Phase 6**: Remove legacy routing and complete migration
- **Phase 7**: Hardening, metrics, and rollback controls

Each phase below includes:

- objective
- implementation steps
- files to touch
- acceptance criteria
- test checklist
- rollback strategy

---

# Phase 0 — Stabilization and Observability

## Objective

Before changing behavior, create enough visibility so the refactor can be measured safely.

## Deliverables

- request-level logs for route selection
- logs for planner outputs
- logs for KB retrieval scores
- logs for search fallback decisions
- logs for state machine transitions
- feature flags for new behavior

## Implementation steps

### Step 0.1 — Add feature flags

Add settings for:

- `ENABLE_KB_CHAT_REFACTOR`
- `ENABLE_COSINE_SEARCH_FALLBACK`
- `ENABLE_UNIFIED_CHAT_RESPONSE`
- `ENABLE_LEGACY_ROUTER`
- `KB_SEARCH_SCORE_THRESHOLD` default `0.5`

Recommended location:

- `core/config.py`

### Step 0.2 — Add request correlation IDs

Ensure each request has:

- `session_id`
- `request_id`
- `route_decision_id` or equivalent

This will make it easier to compare old and new flows.

### Step 0.3 — Add structured logs

At minimum, log:

- normalized query
- rewritten query
- multi-intent status
- subqueries
- top KB score
- threshold used
- whether search fallback was triggered
- current sales state
- next sales state

### Step 0.4 — Add shadow-mode support

When the feature flag is enabled, the system should be able to:

- run the **new planner** in parallel
- not affect the final response yet
- only log differences

This is the safest way to start.

## Files to touch

- `service/RAG/core/config.py`
- `service/RAG/api/routes_sales.py`
- `service/RAG/rag/retriever.py`
- shared logging helpers if any

## Acceptance criteria

- All new logs appear in dev/staging
- No change in user-facing behavior yet
- Feature flags can toggle the refactor path off instantly

## Rollback

- Disable the feature flags
- Revert logging-only changes if needed

---

# Phase 1 — Extract the Planner from `plan_query_adaptive()`

## Objective

Preserve the valuable parts of the existing planner:

- query rewriting
- multi-intent detection
- subquery decomposition

But stop using it as a full top-level route classifier for `RAG | SEARCH | OTHER`.

## Design decision

Refactor `plan_query_adaptive()` into a new planner that produces:

- normalized query
- rewritten query
- multi-intent decision
- decomposed subqueries
- optional intent hints

But **not** the final user-facing route.

## New contract

Create a planner output model like:

```python
class KnowledgePlan(TypedDict, total=False):
    original_query: str
    normalized_query: str
    rewritten_query: str
    multi_intent: bool
    subqueries: list[dict]
    planner_notes: dict
```

Each subquery should look like:

```python
{
  "query": "...",
  "intent_hint": "project_info|comparison|objection|generic|unknown",
  "source_preference": "kb_first"
}
```

Do not include `route = RAG|SEARCH|OTHER` here anymore.

## Implementation steps

### Step 1.1 — Introduce a new planner module

Create a new module, for example:

- `service/RAG/knowledge_base/planner.py`

Move or wrap logic from:

- `plan_query_adaptive()`
- `decompose_subqueries()`
- normalization helpers

### Step 1.2 — Keep the current planner behavior compatible

At first, keep the same LLM prompt shape and decomposition logic, but change the return schema.

### Step 1.3 — Add planner unit tests

Test cases:

1. single-intent product query
2. comparison query
3. objection query
4. greeting
5. mixed query with two independent intents
6. vague follow-up that depends on chat history

### Step 1.4 — Preserve multi-intent behavior

Rules to keep:

- only decompose when the user has truly independent asks
- maximum 2 subqueries
- do not invent missing facts
- keep rewritten subqueries faithful to the original meaning

### Step 1.5 — Define deterministic fallback

If LLM planner fails:

- use the original query as `rewritten_query`
- set `multi_intent = False`
- set `subqueries = []`

## Files to touch

- `service/RAG/rag/retriever.py`
- `service/RAG/knowledge_base/planner.py` (new)
- tests

## Acceptance criteria

- planner still rewrites queries
- planner still supports multi-intent
- planner no longer decides top-level `SEARCH` vs `OTHER`
- old code path remains available behind a flag

## Test checklist

- planner output is JSON-safe
- planner returns consistent schema on failures
- planner preserves query meaning
- decomposition does not split simple questions unnecessarily

## Rollback

- switch back to current `plan_query_adaptive()` path using feature flag

---

# Phase 2 — Add Cosine-Based KB Confidence Evaluation

## Objective

Decide whether internal KB retrieval is sufficient before triggering external search.

## Design decision

After KB retrieval, compute a confidence score using the top retrieved result.

If the score is high enough:

- trust KB-only mode

If the score is lower than threshold:

- trigger search fallback

## Important note

Do **not** rely only on an LLM for deciding whether search is needed.

Use retrieval evidence first.

## Recommended scoring behavior

Use:

- top-1 similarity score
- optionally average of top-3
- optionally score gap between top-1 and top-2

Minimum stable starting rule:

```python
use_search_fallback = top_score < 0.5
```

## Edge-case policy

Some queries should bypass cosine logic or receive special handling:

### Always prefer search-like fallback for

- weather
- time / current hour in a place
- latest news
- exchange rate
- gold price
- stock price
- “today”, “now”, “currently”, “latest”

### Always prefer KB-first for

- project name + internal product facts
- apartment type
- legal / policy / offer already in internal docs
- inventory or project-specific comparison

### Very short vague queries

Examples:

- “price?”
- “which one is better?”
- “how about that project?”

These should usually go to `chat` first unless:
- the session has enough prior context to resolve them

## Implementation steps

### Step 2.1 — Create a KB retrieval evaluator

Create:

- `service/RAG/knowledge_base/evaluator.py`

Recommended output:

```python
class KnowledgeDecision(TypedDict, total=False):
    top_score: float
    avg_top_k_score: float
    should_search: bool
    decision_reason: str
    matched_documents: list[dict]
    low_confidence: bool
    special_case_triggered: bool
```

### Step 2.2 — Expose retrieval scores from the retriever

Make sure the internal retriever returns:

- raw candidate list
- similarity scores
- payload metadata

The current `query_rag()` is answer-oriented. You need a lower-level retrieval function for scoring.

Recommended new function:

```python
async def retrieve_kb_candidates(
    query: str,
    top_k: int = 6,
) -> list[dict]:
    ...
```

Each candidate should include:

```python
{
  "score": 0.81,
  "content": "...",
  "source": "...",
  "document_id": "...",
  "metadata": {...}
}
```

### Step 2.3 — Add a special-case detector

Create deterministic helpers such as:

- `is_realtime_query(text)`
- `is_short_ambiguous_query(text, history)`
- `is_project_specific_query(text, history)`

These should run **before** or **alongside** cosine threshold logic.

### Step 2.4 — Add search fallback orchestrator

Create a function like:

```python
async def build_knowledge_payload(plan, history, state) -> dict:
    ...
```

Behavior:

1. planner rewrites query
2. retriever fetches KB candidates
3. evaluator scores confidence
4. if low confidence and not blocked, call search
5. package both KB and search evidence cleanly

### Step 2.5 — Keep search and KB facts separated

Do not merge search text directly into KB text.

Keep this structure:

```python
{
  "kb_evidence": [...],
  "search_evidence": [...],
  "decision": {...}
}
```

## Files to touch

- `service/RAG/rag/retriever.py`
- `service/RAG/knowledge_base/evaluator.py` (new)
- `service/RAG/knowledge_base/service.py` (new)
- search wrapper module if needed

## Acceptance criteria

- top similarity score is available in logs
- search fallback triggers only when intended
- realtime questions do not get stuck in KB-only mode
- project-specific queries do not unnecessarily go to search

## Test checklist

- high-score KB query -> no search
- low-score KB query -> search enabled
- weather query -> special case triggers search
- short ambiguous query -> stays in chat if unresolved
- known project query -> KB-first behavior preserved

## Rollback

- disable `ENABLE_COSINE_SEARCH_FALLBACK`
- keep KB-only or legacy route logic

---

# Phase 3 — Define the Payload Between `knowledge_base` and `chat`

## Objective

Design a clean, explicit contract so `chat` knows exactly what kind of evidence it received.

This is critical to avoid mixing:

- internal facts
- external search facts
- unresolved ambiguity
- recommendation context

## Design rule

The payload between `knowledge_base` and `chat` must be:

- structured
- typed
- source-aware
- easy to log
- stable across phases

## Recommended payload

Create a typed model in `models/api_models.py` or a dedicated models module.

Example:

```python
class KnowledgeEvidenceItem(BaseModel):
    source_type: str   # "kb" | "search"
    content: str
    score: float | None = None
    source_name: str | None = None
    document_id: str | None = None
    metadata: dict = Field(default_factory=dict)

class KnowledgeDecisionPayload(BaseModel):
    original_query: str
    rewritten_query: str
    multi_intent: bool = False
    subqueries: list[dict] = Field(default_factory=list)
    top_score: float | None = None
    threshold: float | None = None
    should_search: bool = False
    decision_reason: str | None = None
    kb_evidence: list[KnowledgeEvidenceItem] = Field(default_factory=list)
    search_evidence: list[KnowledgeEvidenceItem] = Field(default_factory=list)
    unresolved: bool = False
```

Then pass this into `chat`.

## Minimum rules for the payload

1. `kb_evidence` must contain only internal KB-derived content
2. `search_evidence` must contain only external results
3. `decision_reason` must explain why search happened or did not happen
4. `unresolved = True` when the query is still too vague for confident answering
5. `multi_intent = True` only when subqueries are actually present

## Implementation steps

### Step 3.1 — Add typed payload models

Put these in:

- `service/RAG/models/api_models.py`
  or
- `service/RAG/models/knowledge_models.py`

### Step 3.2 — Make `knowledge_base` return this payload

Create a service method like:

```python
async def resolve_knowledge(
    query: str,
    history: list[dict],
    session_context: dict | None = None,
) -> KnowledgeDecisionPayload:
    ...
```

### Step 3.3 — Add source provenance

Every evidence item should contain enough provenance for debugging:

- KB document ID or file name
- search domain or source URL
- retrieval score if available

### Step 3.4 — Add payload sanitizer

Before the payload enters `chat`, sanitize:

- excessively long evidence blocks
- duplicate snippets
- malformed metadata
- empty entries

## Files to touch

- `service/RAG/models/api_models.py`
- `service/RAG/knowledge_base/service.py`
- `service/RAG/knowledge_base/types.py` if preferred

## Acceptance criteria

- payload schema is explicit and stable
- `chat` can tell KB and search evidence apart
- logs clearly show what evidence entered chat

## Test checklist

- KB-only payload
- search-only payload
- mixed payload
- unresolved payload
- multi-intent payload with two subqueries

## Rollback

- adapt old `query_rag()` string output into temporary payload wrapper

---

# Phase 4 — Make `chat` the Only User-Facing Response Layer

## Objective

Ensure that all final responses come from one place: `chat`.

This avoids having:
- direct RAG answers
- direct search summaries
- free-form OTHER responses
- sales responses
all behaving differently.

## Design rule

Even when the query is fully answerable from KB, the result should still go through `chat`.

That allows:

- consistent tone
- personalization
- state-aware follow-up
- smoother recommendation flow
- better comparison handling

## Implementation steps

### Step 4.1 — Create a unified chat service

Create a single entry such as:

```python
async def run_chat_route(
    session_id: str,
    message: str,
    knowledge_payload: KnowledgeDecisionPayload | None,
    raw_transcript: str | None = None,
) -> dict:
    ...
```

### Step 4.2 — Replace direct response paths

Refactor the current route logic so that:

- `knowledge_base` returns evidence only
- `chat` renders the final response

Remove direct user-facing responses from:

- `_run_rag_flow()`
- `_run_search_flow()`
- `_run_other_llm_flow()`

These can still exist temporarily as internal adapters during migration.

### Step 4.3 — Preserve warm conversational behavior

`chat` should:
- answer naturally
- ask one follow-up question when useful
- keep the discussion around:
  - project
  - apartment preferences
  - lifestyle
  - environment
  - family needs
  - budget and location
- avoid sounding robotic or transactional

### Step 4.4 — Add response modes

The chat layer should be able to handle these modes:

- greeting
- discovery
- recommendation
- project QA
- comparison
- objection handling
- next-step closing

These modes can still be driven by the state machine.

### Step 4.5 — Add internal prompt sections

The prompt should clearly separate:

- user message
- relevant chat history summary
- lead profile
- sales state
- missing slots
- KB evidence
- search evidence
- instruction on what to do when evidence is missing

## Files to touch

- `service/RAG/api/routes_sales.py`
- `service/RAG/sales/graph.py`
- `service/RAG/sales/nodes/build_response.py`
- prompt builder modules
- new chat service module if needed

## Acceptance criteria

- all user-facing responses come through one chat path
- no direct KB summary is returned to the user outside chat
- tone is consistent across greeting, QA, and recommendation

## Test checklist

- greeting without KB
- KB-backed factual answer
- search-backed answer
- mixed KB + search answer
- vague comparison requiring a clarifying question
- user asks a project question after a long conversation

## Rollback

- keep a flag to use legacy `_run_rag_flow()` or `_run_other_llm_flow()` if needed

---

# Phase 5 — Preserve and Adapt the Sales State Machine

## Objective

Keep the deterministic sales orchestration while allowing the new two-route architecture.

## Important principle

Do **not** collapse the state machine into a fully free-form chat agent.

The state machine is valuable because it makes transitions deterministic and auditable.

## Existing baseline

The current graph already has a useful structure:

- ingest user turn
- parse / classify
- update lead profile
- resolve sales state
- resolve script step
- decide response action
- retrieve context if needed
- build response
- validate
- persist
- finalize

This is a strong foundation and should be reused.

## Recommended approach

Keep the state machine in `chat`, but adapt it so it consumes the new `knowledge_payload`.

### New rule

`knowledge_base` answers the question:
> “What evidence do we have?”

The state machine answers the question:
> “What should we do next in the conversation?”

## Implementation steps

### Step 5.1 — Extend graph state

Add fields to `SalesAgentState` such as:

```python
knowledge_payload: Optional[Dict[str, Any]]
knowledge_decision_reason: Optional[str]
kb_top_score: Optional[float]
search_used: Optional[bool]
```

### Step 5.2 — Feed knowledge payload into the graph

When calling the chat graph, include:

- planner result
- KB evidence
- search evidence
- confidence decision

### Step 5.3 — Update `resolve_next_state()`

Keep state transitions deterministic.

Use the new payload only as **supporting context**, not as the primary state engine.

Examples:

- if user is asking recommendation and required slots are missing -> `need_discovery`
- if user is asking project facts and KB evidence exists -> `project_qa`
- if user asks comparison and there are two evidence blocks -> `comparison`
- if evidence is weak and user intent is vague -> remain in `need_discovery` or `greeting`

### Step 5.4 — Update response action logic

Make `response_action` explicit based on both:
- sales state
- knowledge payload quality

Example actions:

- `ask_follow_up`
- `answer_with_kb`
- `answer_with_search`
- `answer_with_kb_and_search`
- `recommend_products`
- `compare_options`
- `handle_objection`
- `invite_next_step`

### Step 5.5 — Keep slot filling deterministic

Continue storing extracted slots like:

- purpose
- location_preference
- property_type
- budget
- family size
- child-related needs
- living environment preferences

Do not rely on open-ended chat memory alone for slot management.

### Step 5.6 — Add “insufficient evidence” behavior

If evidence is weak, the agent should not hallucinate. It should either:

- ask a focused follow-up question
- explain the limitation briefly
- move the conversation toward a clarifying step

## Files to touch

- `service/RAG/sales/graph_state.py`
- `service/RAG/sales/state_machine.py`
- `service/RAG/sales/nodes/resolve_sales_state.py`
- `service/RAG/sales/nodes/decide_response_action.py`
- `service/RAG/sales/nodes/build_response.py`

## Acceptance criteria

- state transitions remain code-driven
- chat uses knowledge payload without losing deterministic flow
- the model knows when to ask, answer, compare, or recommend

## Test checklist

- missing slots -> discovery
- enough slots + KB evidence -> product matching
- project question + exact KB evidence -> project QA
- vague comparison + weak evidence -> clarifying question
- objection + evidence available -> objection handling

## Rollback

- keep previous state machine untouched behind a feature flag until migration is complete

---

# Phase 6 — Migrate API Contracts and Remove Legacy Top-Level Routing

## Objective

Move the top-level API surface to the new two-route model without breaking clients.

## Recommended endpoint behavior

Keep the external API simple:

- `/sales/chat`
- `/sales/chat/stream`
- `/query/stream` as backward-compatible alias if needed

Internally:

- `knowledge_base` is a service
- `chat` is the main orchestrator

## Implementation steps

### Step 6.1 — Add internal service boundaries

Create:

- `knowledge_base/service.py`
- `chat/service.py`

### Step 6.2 — Refactor `routes_sales.py`

Replace `_run_sales_or_search()` with:

1. build `knowledge_payload` when needed
2. call unified chat route
3. return unified response format

### Step 6.3 — Update response contract

Add fields to the response for debugging and observability:

```python
class SalesChatResponse(BaseModel):
    session_id: str
    response: str
    route_category: Optional[str] = None      # "CHAT"
    sales_state: Optional[str] = None
    lead_profile: Optional[Dict[str, Any]] = None
    missing_slots: Optional[List[str]] = None
    knowledge_used: bool = False
    search_used: bool = False
    kb_top_score: Optional[float] = None
```

### Step 6.4 — Keep streaming contract stable

Streaming responses should remain backward-compatible for frontend stability.

Do not change:
- NDJSON framing
- phase keys
- session_id behavior

Unless the frontend is updated together.

## Acceptance criteria

- clients can continue calling the same endpoints
- internal routing is simplified
- response metadata reflects the new architecture

## Rollback

- restore `_run_sales_or_search()` as the main internal path

---

# Phase 7 — Hardening, Evaluation, and Production Rollout

## Objective

Safely move from feature-flagged rollout to production default.

## Metrics to track

### Quality metrics

- recommendation relevance
- hallucination rate
- follow-up question quality
- user engagement depth
- conversion-oriented behavior

### Retrieval metrics

- top KB score distribution
- search fallback rate
- special-case trigger rate
- unresolved query rate

### State-machine metrics

- greeting -> discovery transition rate
- discovery -> product matching transition rate
- comparison and objection path usage
- fallback-to-clarification rate

### Performance metrics

- planner latency
- KB retrieval latency
- search latency
- chat response latency
- total end-to-end latency

## Rollout plan

### Stage 1 — Log-only shadow mode
- new planner runs in parallel
- no user-visible changes

### Stage 2 — Internal dev testing
- enable for test sessions only

### Stage 3 — Limited staging rollout
- enable for a subset of traffic

### Stage 4 — Production with kill switch
- enable default path
- preserve legacy fallback for emergencies

### Stage 5 — Remove dead code
- only after stable operation and metrics confirm quality

---

## 5. Suggested New Module Layout

```text
service/RAG/
├── api/
│   └── routes_sales.py
├── knowledge_base/
│   ├── planner.py
│   ├── evaluator.py
│   ├── service.py
│   └── special_cases.py
├── chat/
│   └── service.py
├── sales/
│   ├── graph.py
│   ├── graph_state.py
│   ├── state_machine.py
│   └── nodes/
├── models/
│   ├── api_models.py
│   └── knowledge_models.py   # optional
└── rag/
    └── retriever.py
```

---

## 6. Recommended PR Breakdown

To keep the rollout stable, use small PRs.

### PR 1
- logging
- flags
- shadow mode scaffolding

### PR 2
- extract planner
- add planner tests

### PR 3
- low-level KB retrieval with scores
- cosine evaluator
- special-case detection

### PR 4
- structured knowledge payload models
- service wrapper

### PR 5
- unify final response through chat
- preserve legacy path behind flag

### PR 6
- wire payload into state machine
- update response action logic

### PR 7
- migrate streaming and API response metadata

### PR 8
- cleanup and remove dead legacy routing

---

## 7. Detailed Testing Matrix

## 7.1 Planner tests

- greeting
- short product query
- multi-intent comparison
- follow-up dependent on history
- malformed planner JSON output
- planner timeout fallback

## 7.2 KB confidence tests

- exact project query with strong KB hit
- weak KB hit with search fallback
- realtime query
- short ambiguous query
- project-specific question with historical session context

## 7.3 Chat behavior tests

- greeting with no KB
- discovery flow with missing slots
- recommendation flow with enough slots
- project QA using KB evidence
- comparison using mixed evidence
- objection handling
- next-step closing invitation

## 7.4 Regression tests

- `/sales/chat`
- `/sales/chat/stream`
- `/query/stream`
- session persistence
- lead profile updates
- pronoun handling
- history continuity

---

## 8. Non-Negotiable Guardrails

1. `knowledge_base` must never directly produce the final user-facing answer
2. `chat` must always know whether evidence came from:
   - internal KB
   - external search
3. the state machine must remain code-driven
4. low-confidence retrieval must not cause hallucinated product recommendations
5. ambiguous short queries must prefer clarification over forced answering
6. multi-intent behavior must be preserved or explicitly downgraded with safe fallback
7. feature flags must allow immediate rollback

---

## 9. Recommended First Implementation Order

If the team wants the safest possible order, implement in this exact sequence:

1. add flags and logs
2. extract planner without behavior change
3. expose KB retrieval scores
4. add cosine evaluator
5. add search fallback behind flag
6. define structured payload
7. route all final responses through chat
8. inject payload into state machine
9. remove legacy top-level routing only after regression passes

---

## 10. Final Recommended End-State

When the refactor is complete, the runtime flow should look like this:

```text
User message
  -> planner
  -> KB retrieval
  -> cosine evaluation
  -> optional web search fallback
  -> structured knowledge payload
  -> chat state machine
  -> final user-facing response
  -> persist memory/history/profile
```

And conceptually:

```text
knowledge_base = evidence builder
chat           = conversational strategist + response renderer
```

That division gives you:

- lower routing complexity
- cleaner control flow
- better explainability
- better observability
- safer fallback behavior
- stronger sales conversation continuity

---

## 11. Recommended Definition of Done

This refactor is done only when all of the following are true:

- planner rewrite and multi-intent still work
- KB similarity scoring is available and trusted
- search fallback is deterministic and logged
- chat is the only user-facing response layer
- sales state transitions remain deterministic
- KB and search evidence are clearly separated
- latency is improved or at least not worsened
- legacy route logic can be removed safely

---

## 12. Optional Next-Step Enhancements

After the main refactor is stable, consider:

- confidence-aware prompt templates
- evidence citation snippets in internal debug logs
- adaptive threshold by query type
- session-aware ambiguity resolver
- reranker for KB evidence before thresholding
- analytics dashboard for state transitions and fallback rates
