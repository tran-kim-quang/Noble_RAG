"""
Prompt optimization module for latency reduction.

Implements:
1. Semantic caching - cache responses based on normalized prompt
2. Prompt compression - reduce prompt token count
3. Streaming support - return responses progressively
4. Timing instrumentation - track per-stage latency
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable, Optional
from datetime import datetime, timedelta
import logging

log = logging.getLogger("sales-orchestrator-optimization")


class SynthesisCacheEntry:
    """Cache entry with TTL and metadata."""
    
    def __init__(self, response: str, metadata: dict[str, Any] | None = None, ttl_seconds: int = 3600):
        self.response = response
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
    
    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def record_hit(self) -> None:
        self.hit_count += 1


class SemanticSynthesisCache:
    """
    Semantic cache for synthesis responses.
    
    Caches based on normalized prompt hash (ignoring minor variations).
    Reduces redundant API calls for similar queries.
    """
    
    def __init__(self, max_entries: int = 1000, default_ttl_seconds: int = 3600):
        self.cache: dict[str, SynthesisCacheEntry] = {}
        self.max_entries = max_entries
        self.default_ttl_seconds = default_ttl_seconds
        self.hits = 0
        self.misses = 0
    
    @staticmethod
    def _normalize_for_hashing(prompt: str) -> str:
        """Normalize prompt for cache key generation (remove minor whitespace variations)."""
        # Remove extra whitespace, normalize line endings
        normalized = " ".join(prompt.split())
        return normalized.lower()
    
    @staticmethod
    def _compute_hash(normalized_prompt: str) -> str:
        """Compute SHA256 hash of normalized prompt."""
        return hashlib.sha256(normalized_prompt.encode()).hexdigest()[:16]
    
    def get(self, prompt: str) -> Optional[str]:
        """
        Retrieve cached response if available and not expired.
        Returns None if cache miss or expired.
        """
        normalized = self._normalize_for_hashing(prompt)
        cache_key = self._compute_hash(normalized)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if entry.is_expired():
                del self.cache[cache_key]
                self.misses += 1
                return None
            
            entry.record_hit()
            self.hits += 1
            return entry.response
        
        self.misses += 1
        return None
    
    def set(self, prompt: str, response: str, metadata: dict[str, Any] | None = None) -> None:
        """Cache the response for future use."""
        if len(self.cache) >= self.max_entries:
            # Evict oldest entry
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k].created_at
            )
            del self.cache[oldest_key]
        
        normalized = self._normalize_for_hashing(prompt)
        cache_key = self._compute_hash(normalized)
        self.cache[cache_key] = SynthesisCacheEntry(
            response,
            metadata=metadata,
            ttl_seconds=self.default_ttl_seconds
        )
    
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        return {
            "total_entries": len(self.cache),
            "max_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests,
        }
    
    def clear(self) -> None:
        """Clear entire cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0


class PromptCompressor:
    """
    Compress synthesis prompts to reduce token count and latency.
    
    Strategies:
    1. Remove redundant instructions
    2. Shorten field names in JSON payloads
    3. Reduce history depth
    4. Minimize whitespace
    """
    
    @staticmethod
    def compress_state_payload(state_payload: dict[str, Any], aggressive: bool = False) -> dict[str, Any]:
        """
        Compress lead state payload by removing low-impact fields.
        
        Args:
            state_payload: Original state dict
            aggressive: If True, remove more fields (higher compression, lower fidelity)
        
        Returns:
            Compressed state dict
        """
        compressed = {}
        
        # Keep only high-impact fields
        essential_fields = {
            "need_summary",
            "need_topics",
            "painpoint_summary",
            "painpoint_topics",
            "sales_state",
            "engagement_state",
            "next_best_action",
        }
        
        optional_fields = {
            "name",
            "phone_contact",
        }
        
        # Always include essential fields
        for field in essential_fields:
            if field in state_payload:
                compressed[field] = state_payload[field]
        
        # Include optional fields only in non-aggressive mode
        if not aggressive:
            for field in optional_fields:
                if field in state_payload:
                    compressed[field] = state_payload[field]
        
        # Limit topics to top 2 (instead of 4) in aggressive mode
        if aggressive:
            if "need_topics" in compressed:
                compressed["need_topics"] = compressed["need_topics"][:2]
            if "painpoint_topics" in compressed:
                compressed["painpoint_topics"] = compressed["painpoint_topics"][:2]
        
        return compressed
    
    @staticmethod
    def compress_history(history: list[dict[str, str]], max_turns: int = 2) -> list[dict[str, str]]:
        """
        Reduce history to most recent turns only.
        
        Args:
            history: Original history list
            max_turns: Maximum number of turns to keep
        
        Returns:
            Compressed history
        """
        if len(history) <= max_turns:
            return history
        
        # Keep only most recent turns
        return history[-max_turns:]
    
    @staticmethod
    def compress_grounded_context(grounded: dict[str, Any], aggressive: bool = False) -> dict[str, Any]:
        """
        Compress grounded context by reducing card/evidence count.
        
        Args:
            grounded: Original grounded context dict
            aggressive: If True, reduce more aggressively
        
        Returns:
            Compressed grounded context
        """
        compressed = grounded.copy()
        
        # Reduce evidence chunks
        if "evidence_chunks" in compressed and isinstance(compressed["evidence_chunks"], list):
            max_evidence = 1 if aggressive else 2
            compressed["evidence_chunks"] = compressed["evidence_chunks"][:max_evidence]
        
        # Reduce proximity facts
        if "proximity_facts" in compressed and isinstance(compressed["proximity_facts"], list):
            max_proximity = 1 if aggressive else 2
            compressed["proximity_facts"] = compressed["proximity_facts"][:max_proximity]
        
        return compressed


class TimingInstrument:
    """Track per-stage latency for performance analysis."""
    
    def __init__(self):
        self.stages: dict[str, list[float]] = {}
        self.start_times: dict[str, float] = {}
    
    def start(self, stage_name: str) -> None:
        """Mark start of a stage."""
        self.start_times[stage_name] = time.time()
    
    def end(self, stage_name: str) -> float:
        """Mark end of a stage, return elapsed milliseconds."""
        if stage_name not in self.start_times:
            log.warning(f"Stage {stage_name} started but no start time recorded")
            return 0.0
        
        elapsed = (time.time() - self.start_times[stage_name]) * 1000  # Convert to ms
        
        if stage_name not in self.stages:
            self.stages[stage_name] = []
        
        self.stages[stage_name].append(elapsed)
        del self.start_times[stage_name]
        
        return elapsed
    
    def get_stage_stats(self, stage_name: str) -> dict[str, float]:
        """Get statistics for a specific stage."""
        if stage_name not in self.stages or not self.stages[stage_name]:
            return {}
        
        times = self.stages[stage_name]
        return {
            "min_ms": min(times),
            "max_ms": max(times),
            "avg_ms": sum(times) / len(times),
            "latest_ms": times[-1],
            "count": len(times),
        }
    
    def get_all_stats(self) -> dict[str, Any]:
        """Get statistics for all stages."""
        return {
            stage: self.get_stage_stats(stage)
            for stage in self.stages.keys()
        }


# Global instances
_synthesis_cache = None
_timing_instrument = None


def init_optimization(
    enable_cache: bool = True,
    cache_max_entries: int = 1000,
    cache_ttl_seconds: int = 3600,
    enable_timing: bool = True,
) -> None:
    """Initialize optimization components."""
    global _synthesis_cache, _timing_instrument
    
    if enable_cache:
        _synthesis_cache = SemanticSynthesisCache(
            max_entries=cache_max_entries,
            default_ttl_seconds=cache_ttl_seconds
        )
        log.info(f"Synthesis cache enabled (max_entries={cache_max_entries}, ttl={cache_ttl_seconds}s)")
    
    if enable_timing:
        _timing_instrument = TimingInstrument()
        log.info("Timing instrumentation enabled")


def get_synthesis_cache() -> Optional[SemanticSynthesisCache]:
    """Get global synthesis cache instance."""
    return _synthesis_cache


def get_timing_instrument() -> Optional[TimingInstrument]:
    """Get global timing instrument instance."""
    return _timing_instrument
