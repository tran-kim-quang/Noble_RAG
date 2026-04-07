Dưới đây là **PR 1 implementation checklist** cho nhánh `quang-dev`, tập trung vào mục tiêu:

> **biến multi-intent từ planner output thành knowledge execution thật sự theo từng subquery**, trong khi vẫn giữ backward compatibility cho `chat`.

Hiện tại `planner` đã sinh được `multi_intent` và `subqueries` , nhưng `resolve_knowledge()` vẫn chỉ chạy retrieve/evaluate trên một `rewritten_query` duy nhất . Đây là phần PR 1 cần sửa.

---

# PR 1 — Multi-intent execution in `knowledge_base`

## 1. Mục tiêu PR

### Mục tiêu chính

* Khi planner trả về `multi_intent=true`, knowledge layer phải:

  * chạy KB retrieval cho **từng subquery**
  * evaluate score cho **từng subquery**
  * bật search fallback cho **từng subquery nếu cần**
  * merge kết quả thành một `KnowledgeDecisionPayload` tổng

### Mục tiêu phụ

* Không làm vỡ API `/sales/chat`
* Không bắt buộc sửa lớn ở `chat.service` trong PR này
* Giữ payload cũ vẫn dùng được
* Chuẩn bị nền cho PR sau để `chat` consume `subquery_results`

---

## 2. Out of scope của PR 1

PR này **không làm** các việc sau:

* chưa update lead profile
* chưa refactor state machine lớn
* chưa bỏ legacy router trong `retriever.py`
* chưa làm session-aware retrieval sâu
* chưa migrate sang full sales graph

---

## 3. File cần sửa

### Sửa chính

* `service/RAG/knowledge_base/service.py` 
* `service/RAG/models/api_models.py` 

### Có thể sửa nhẹ

* `service/RAG/knowledge_base/planner.py` nếu muốn chuẩn hóa thêm shape của subquery output 

### Không nên đụng nhiều trong PR này

* `chat/service.py`
* `sales/state_machine.py`
* `rag/retriever.py`

---

## 4. Checklist triển khai

## Step 1 — Mở rộng payload model để chứa kết quả theo từng subquery

### Việc cần làm

Trong `models/api_models.py`, thêm model mới cho từng subquery result.

### Nên thêm

```python
class KnowledgeSubqueryResult(BaseModel):
    query: str
    intent_hint: Optional[str] = None
    source_preference: Optional[str] = None
    top_score: Optional[float] = None
    threshold: Optional[float] = None
    should_search: bool = False
    decision_reason: Optional[str] = None
    kb_evidence: List[KnowledgeEvidenceItem] = Field(default_factory=list)
    search_evidence: List[KnowledgeEvidenceItem] = Field(default_factory=list)
    unresolved: bool = False
```

Sau đó mở rộng `KnowledgeDecisionPayload`:

```python
class KnowledgeDecisionPayload(BaseModel):
    ...
    subquery_results: List[KnowledgeSubqueryResult] = Field(default_factory=list)
```

### Điều cần giữ

* không xóa các field hiện tại như:

  * `kb_evidence`
  * `search_evidence`
  * `top_score`
  * `decision_reason`
  * `unresolved` 
* vì `chat.service` hiện vẫn đang đọc payload tổng, chưa đọc `subquery_results`

### Done when

* payload mới parse được
* code cũ vẫn validate được nếu `subquery_results=[]`

---

## Step 2 — Tách pipeline xử lý một subquery thành helper riêng

### Việc cần làm

Trong `knowledge_base/service.py`, tạo helper nội bộ để xử lý **một query**.

### Nên thêm hàm

```python
async def _resolve_single_query(
    *,
    query: str,
    history: Optional[List[Dict[str, Any]]],
    top_k: int,
) -> Dict[str, Any]:
    ...
```

### Hàm này phải làm

1. gọi `retrieve_kb_candidates(query, top_k=top_k)` 
2. gọi `evaluate_kb_candidates(...)` 
3. nếu `should_search=True` và feature flag bật thì gọi `tavily_search(query)` 
4. sanitize:

   * `kb_evidence`
   * `search_evidence`
5. trả về object kiểu:

```python
{
    "query": query,
    "top_score": ...,
    "threshold": ...,
    "should_search": ...,
    "decision_reason": ...,
    "kb_evidence": ...,
    "search_evidence": ...,
    "unresolved": ...,
}
```

### Lưu ý

* dùng lại `_sanitize_evidence(...)`
* dùng lại `_is_unresolved(...)`
* không duplicate logic sanitize trong nhiều chỗ

### Done when

* knowledge logic cho single query nằm gọn trong 1 helper
* `resolve_knowledge()` chỉ còn vai trò orchestration

---

## Step 3 — Thêm orchestration cho multi-intent

### Việc cần làm

Sửa `resolve_knowledge()` để branch theo:

* **single-intent** → giữ behavior gần như cũ
* **multi-intent** → loop qua từng subquery

### Logic mới đề xuất

```python
knowledge_plan = plan or await build_knowledge_plan(query, history)
subqueries = list(knowledge_plan.get("subqueries") or [])

if knowledge_plan.get("multi_intent") and subqueries:
    # process each subquery
else:
    # process rewritten_query only
```

### Với multi-intent

Cho mỗi subquery:

* lấy `query`
* lấy `intent_hint`
* lấy `source_preference`
* gọi `_resolve_single_query(...)`
* build `KnowledgeSubqueryResult`

### Output phải có

* `subquery_results`
* merged `kb_evidence`
* merged `search_evidence`
* merged `should_search`
* merged `unresolved`

### Rule merge tối thiểu

* `should_search = any(subquery.should_search)`
* `unresolved = any(subquery.unresolved)` hoặc true nếu không có evidence nào
* `top_score = max(top_score of subqueries)` hoặc `None` nếu rỗng
* `decision_reason` ở payload tổng có thể là:

  * `"multi_intent_mixed"`
  * hoặc chuỗi gộp ngắn như `"multi_intent: low_kb_confidence + project_specific_kb_first"`

### Done when

* `resolve_knowledge()` trả payload tổng hợp được cho multi-intent
* không còn chỉ retrieve theo một `rewritten_query` duy nhất trong case multi-intent

---

## Step 4 — Merge evidence tổng nhưng phải dedupe

### Việc cần làm

Tạo helper mới để merge evidence từ nhiều subqueries.

### Nên thêm hàm

```python
def _merge_evidence_lists(
    items: List[KnowledgeEvidenceItem],
    *,
    max_items: int = 8,
) -> List[KnowledgeEvidenceItem]:
    ...
```

### Rule dedupe

Dùng key:

* `source_type`
* `source_name`
* normalized `content`

### Rule ordering

Ưu tiên:

1. KB evidence trước
2. score cao trước
3. search evidence sau

### Rule size limit

* tổng `kb_evidence` merged: tối đa 6–8 item
* tổng `search_evidence` merged: tối đa 3–4 item

### Lý do

Nếu không giới hạn, prompt cho chat sẽ bị phình quá nhanh.

### Done when

* payload merged gọn
* không bị lặp cùng một snippet nhiều lần

---

## Step 5 — Giữ backward compatibility cho `chat.service`

### Việc cần làm

Trong PR này, **không bắt buộc** sửa `chat.service` lớn.

### Nhưng cần đảm bảo

`KnowledgeDecisionPayload` sau khi mở rộng vẫn có đầy đủ các field cũ mà `chat.service` đang đọc:

* `kb_evidence`
* `search_evidence`
* `decision_reason`
* `unresolved`
* `top_score`
* `subqueries`
* `multi_intent`  

### Nguyên tắc

* `chat.service` có thể chưa hiểu `subquery_results`
* nhưng payload tổng merged phải đủ để nó vẫn hoạt động như hiện tại

### Done when

* không cần sửa `chat.service` mà `/sales/chat` vẫn chạy

---

## Step 6 — Cập nhật logging để debug multi-intent

### Việc cần làm

Trong `knowledge_base/service.py`, thêm log riêng cho:

* planner result
* số lượng subqueries
* decision từng subquery
* merge result cuối

### Nên log

* `query`
* `rewritten_query`
* `multi_intent`
* `subquery_count`
* với mỗi subquery:

  * `subquery`
  * `top_score`
  * `should_search`
  * `decision_reason`
  * `kb_hit_count`
  * `search_hit_count`
* merged:

  * `total_kb_evidence`
  * `total_search_evidence`
  * `payload_unresolved`

### Done when

* nhìn log là biết từng subquery được xử lý ra sao

---

## 5. Pseudocode target

Bạn có thể refactor `resolve_knowledge()` theo khung này:

```python
async def resolve_knowledge(...):
    knowledge_plan = plan or await build_knowledge_plan(query, history)
    rewritten_query = ...
    subqueries = ...

    if knowledge_plan.get("multi_intent") and subqueries:
        subquery_results = []
        merged_kb = []
        merged_search = []

        for item in subqueries:
            sq_query = item["query"]
            single = await _resolve_single_query(
                query=sq_query,
                history=history,
                top_k=top_k,
            )
            subquery_results.append(
                KnowledgeSubqueryResult(
                    query=sq_query,
                    intent_hint=item.get("intent_hint"),
                    source_preference=item.get("source_preference"),
                    top_score=single["top_score"],
                    threshold=single["threshold"],
                    should_search=single["should_search"],
                    decision_reason=single["decision_reason"],
                    kb_evidence=single["kb_evidence"],
                    search_evidence=single["search_evidence"],
                    unresolved=single["unresolved"],
                )
            )
            merged_kb.extend(single["kb_evidence"])
            merged_search.extend(single["search_evidence"])

        payload = KnowledgeDecisionPayload(
            original_query=query,
            rewritten_query=rewritten_query,
            multi_intent=True,
            subqueries=subqueries,
            subquery_results=subquery_results,
            kb_evidence=_merge_evidence_lists(merged_kb),
            search_evidence=_merge_evidence_lists(merged_search),
            top_score=max(...),
            threshold=float(settings.kb_search_score_threshold),
            should_search=any(...),
            decision_reason="multi_intent_mixed",
            unresolved=...,
            planner_notes=...,
        )
        return payload

    single = await _resolve_single_query(...)
    return KnowledgeDecisionPayload(...)
```

---

## 6. Test checklist

## Unit tests cần có

### A. Single-intent regression

* [ ] query đơn vẫn cho payload giống behavior cũ
* [ ] `subquery_results=[]`
* [ ] `kb_evidence` và `search_evidence` vẫn đúng

### B. Multi-intent with both subqueries confident in KB

Ví dụ:

* “so sánh pháp lý dự án A và chính sách bán hàng dự án B”

Expected:

* [ ] `multi_intent=True`
* [ ] có 2 `subquery_results`
* [ ] cả 2 subquery không cần search
* [ ] merged `kb_evidence` có dữ liệu từ cả hai subquery

### C. Mixed confidence

Ví dụ:

* subquery A là project info nội bộ
* subquery B là thị trường / realtime

Expected:

* [ ] subquery A → `should_search=False`
* [ ] subquery B → `should_search=True`
* [ ] merged payload có cả `kb_evidence` và `search_evidence`

### D. Multi-intent but one subquery unresolved

Expected:

* [ ] `subquery_results[i].unresolved=True` cho subquery mơ hồ
* [ ] payload tổng vẫn hợp lệ
* [ ] merged evidence không bị rỗng toàn bộ nếu subquery còn lại có data

### E. Deduplication

Expected:

* [ ] cùng một snippet KB xuất hiện ở 2 subquery chỉ giữ 1 lần trong merged payload

### F. Planner says multi_intent=true but subqueries malformed

Expected:

* [ ] fallback an toàn
* [ ] không crash
* [ ] payload vẫn hợp lệ

---

## 7. Definition of Done

PR 1 được coi là xong khi:

* [ ] `resolve_knowledge()` xử lý thật sự theo từng subquery khi `multi_intent=true`
* [ ] có `subquery_results` trong payload
* [ ] merged payload vẫn backward-compatible với `chat.service`
* [ ] single-intent không bị regression
* [ ] log đủ để debug từng subquery
* [ ] test pass cho:

  * single-intent
  * multi-intent
  * mixed KB/search
  * unresolved subquery
  * dedupe behavior

---

## 8. Commit breakdown gợi ý

### Commit 1

`add subquery result models to knowledge payload`

* sửa `api_models.py`

### Commit 2

`extract single-query knowledge resolution helper`

* thêm `_resolve_single_query()` trong `knowledge_base/service.py`

### Commit 3

`add multi-intent orchestration and merged payload building`

* loop qua subqueries
* build `subquery_results`
* merge evidence

### Commit 4

`add multi-intent logging and tests`

* log
* unit tests
* regression tests

---

## 9. File ưu tiên sửa đầu tiên

Bắt đầu theo đúng thứ tự này:

1. `service/RAG/models/api_models.py` 
2. `service/RAG/knowledge_base/service.py` 
3. test files cho knowledge service
4. chỉ sửa `planner.py` nếu cần normalize thêm shape output 

---

## 10. Kết luận

PR 1 nên được giữ rất rõ scope:

**Không làm chat thông minh hơn ngay.**
**Không làm state machine phức tạp hơn ngay.**
**Chỉ làm knowledge layer xử lý multi-intent thật.**

Đây là nền để các PR sau:

* cho `chat` consume `subquery_results`
* cho state machine hiểu comparison/recommendation tốt hơn
* cho profile enrichment hoạt động đúng ngữ cảnh

