# So do Mermaid - Luong moi hien tai cua system agent

Tai lieu nay mo ta luong online hien tai theo code trong `orchestrator_service` + `retrieval_service`.

## 1) Kien truc tong the (component)

```mermaid
flowchart LR
  U[User / Client] -->|POST /sales/query| O[Orchestrator Service\nFastAPI :8021]

  subgraph ORCH[orchestrator_service/app.py]
    O --> A[Turn Analyzer\nLLM decider]
    A --> R1{start_route}
    R1 -->|consult_discovery| C[run_consult_discovery\nneed/painpoint delta + routing_signal]
    R1 -->|project_grounded| G[run_project_grounded]

    C --> CH{routing_signal.should_route_project}
    CH -->|true| G
    CH -->|false| S1[Merge lead_state]

    G --> RC[RetrievalClient\n/retrieve/project-grounded]
    RC --> RS[Retrieval Service :8011]
    RS --> E[Embedding Endpoint\nremote/local]
    RS --> Q[(Qdrant)]

    G --> S1
    S1 --> SS[Sales State Engine\nupdate_sales_state\nselect_conversation_goal\nselect_next_best_action]
    SS --> RP[Reply Planner\nresponse_mode + ask_policy]
    RP --> SYN[synthesize_assistant_reply\nLLM JSON + sanitize/rewrite/fallback]
    SYN --> RESP[QueryResponse\nroute + assistant_reply + decision_trace\n+ optional project_grounded_payload]
  end

  RESP --> U

  A -. decider_api_url .-> M1[(LLM Generate API\nORCHESTRATOR_DECIDER_API_URL)]
  SYN -. decider_api_url .-> M1
  RS -. EMBEDDING_API_URL .-> M2[(Embedding API\nEMBEDDING_API_URL)]
  RS -. optional warm loop .-> M1
  RS -. optional warm loop .-> M2
```

## 2) Sequence luong online /sales/query

```mermaid
sequenceDiagram
  autonumber
  participant Client
  participant Orch as Orchestrator /sales/query
  participant Decider as LLM Decider API
  participant Ret as Retrieval Service
  participant Emb as Embedding API
  participant Q as Qdrant

  Client->>Orch: POST /sales/query {message, lead_state, recent_history}

  Orch->>Decider: analyze_turn (JSON prompt)
  alt Decider OK
    Decider-->>Orch: query_type, start_route, should_route_project, deltas, hints
  else Decider timeout/error
    Orch-->>Orch: fallback analyze_turn (consult_discovery)
  end

  alt start_route = consult_discovery
    Orch-->>Orch: run_consult_discovery
    alt should_route_project = true
      Orch->>Ret: POST /retrieve/project-grounded
    else should_route_project = false
      Orch-->>Orch: skip grounded retrieval
    end
  else start_route = project_grounded
    Orch->>Ret: POST /retrieve/project-grounded
  end

  opt grounded retrieval executed
    Ret->>Emb: embed candidate/proximity/evidence queries
    Ret->>Q: multi-pass vector retrieval
    Q-->>Ret: documents + scores
    Ret-->>Orch: project_cards + trait_tags + proximity_facts + evidence_chunks + confidence
  end

  Orch-->>Orch: merge lead_state
  Orch-->>Orch: update sales_state + conversation_goal + next_best_action
  Orch->>Decider: synthesize_assistant_reply (JSON)
  Decider-->>Orch: assistant_reply
  Orch-->>Orch: sanitize/rewrite/fallback by ask_policy

  Orch-->>Client: QueryResponse (route, assistant_reply, decision_trace, optional grounded payload)
```

## 3) Ket qua check runtime nhanh (thoi diem hien tai)

- `GET http://127.0.0.1:8021/health` -> `ok`
- `GET http://127.0.0.1:8011/health` (tu host) -> connection refused
- `POST /sales/query` test nhe bi timeout ~45s (giong timeout decider hien tai)

=> Luong code da dung theo kien truc tren; runtime hien tai dang co nghen/phu thuoc o dependency (decider/retrieval endpoint).
