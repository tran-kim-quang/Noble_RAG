"""LangGraph Sales Orchestrator.

Defines and compiles the StateGraph for the AI Sales Agent.

Pipeline (optimized):
  ingest_user_turn
    → classify_and_extract   ← 1 LLM call (was: classify_intent + extract_lead_slots = 2 calls)
    → update_lead_profile
    → resolve_sales_state
    → [retrieve_context →] build_response
    → persist_turn → finalize_output
"""

from langgraph.graph import StateGraph, END

from sales.graph_state import SalesAgentState
from sales.edges import route_after_state_resolution
from sales.nodes.ingest_user_turn import ingest_user_turn
from sales.nodes.classify_and_extract import classify_and_extract
from sales.nodes.update_lead_profile import update_lead_profile
from sales.nodes.resolve_sales_state import resolve_sales_state_node
from sales.nodes.retrieve_context import retrieve_context
from sales.nodes.build_response import build_response
from sales.nodes.persist_turn import persist_turn
from sales.nodes.finalize_output import finalize_output

# ── Build graph ────────────────────────────────────────────────────────────
builder = StateGraph(SalesAgentState)

builder.add_node("ingest_user_turn", ingest_user_turn)
builder.add_node("classify_and_extract", classify_and_extract)
builder.add_node("update_lead_profile", update_lead_profile)
builder.add_node("resolve_sales_state", resolve_sales_state_node)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("build_response", build_response)
builder.add_node("persist_turn", persist_turn)
builder.add_node("finalize_output", finalize_output)

# ── Entry point ────────────────────────────────────────────────────────────
builder.set_entry_point("ingest_user_turn")

# ── Linear edges ──────────────────────────────────────────────────────────
builder.add_edge("ingest_user_turn", "classify_and_extract")
builder.add_edge("classify_and_extract", "update_lead_profile")
builder.add_edge("update_lead_profile", "resolve_sales_state")

# ── Conditional routing after state resolution ────────────────────────────
builder.add_conditional_edges(
    "resolve_sales_state",
    route_after_state_resolution,
    {
        "retrieve_context": "retrieve_context",
        "build_response": "build_response",
    },
)

builder.add_edge("retrieve_context", "build_response")
builder.add_edge("build_response", "persist_turn")
builder.add_edge("persist_turn", "finalize_output")
builder.add_edge("finalize_output", END)

# ── Compile ────────────────────────────────────────────────────────────────
sales_graph = builder.compile()
