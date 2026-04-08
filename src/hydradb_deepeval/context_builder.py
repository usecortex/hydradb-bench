"""
Build a formatted context string from HydraDB full_recall response.

Python port of the TypeScript buildContextString() function.
"""

from __future__ import annotations

from typing import Any


def _format_path_chain(path: Any) -> str:
    """Format a ScoredPathResponse into a readable chain string.

    Mirrors TypeScript formatPathChain(path: ScoredPathResponse):
      path.triplets.map(t => "[src] -> predicate -> [tgt]: context [Time: ...]").join("\n  ↳ ")
    """
    if not isinstance(path, dict):
        return str(path)

    triplets = path.get("triplets") or []
    parts = []
    for t in triplets:
        src = (t.get("source") or {}).get("name", "")
        tgt = (t.get("target") or {}).get("name", "")
        rel = t.get("relation") or {}
        pred = rel.get("canonical_predicate", "")
        line = f"[{src}] -> {pred} -> [{tgt}]"
        ctx = rel.get("context")
        if ctx:
            line += f": {ctx}"
        temporal = rel.get("temporal_details")
        if temporal:
            line += f" [Time: {temporal}]"
        parts.append(line)

    return "\n  -> ".join(parts)


def build_context_string(result: dict[str, Any]) -> str:
    """
    Build a formatted context string from a HydraDB full_recall response dict.

    Mirrors the TypeScript buildContextString() function:
      - Entity paths from graph_context.query_paths
      - Chunks with source title, content, graph relations, extra context
    """
    lines: list[str] = []
    gc: dict[str, Any] = result.get("graph_context") or {}

    # ── Entity paths ──────────────────────────────────────────────────────
    query_paths = gc.get("query_paths") or []
    if query_paths:
        lines.append("=== ENTITY PATHS ===")
        for path in query_paths:
            lines.append(_format_path_chain(path))
        lines.append("")

    # ── Chunks ────────────────────────────────────────────────────────────
    chunks: list[dict[str, Any]] = result.get("chunks") or []
    additional_context: dict[str, Any] = result.get("additional_context") or {}
    chunk_id_to_group_ids: dict[str, list[str]] = gc.get("chunk_id_to_group_ids") or {}
    chunk_relations: list[dict[str, Any]] = gc.get("chunk_relations") or []

    if chunks:
        lines.append("=== CONTEXT ===")
        for i, chunk in enumerate(chunks):
            lines.append(f"Chunk {i + 1}")
            source = chunk.get("source_title") or chunk.get("source") or ""
            if source:
                lines.append(f"Source: {source}")
            lines.append(chunk.get("chunk_content") or chunk.get("content") or "")

            # Graph relations for this chunk
            chunk_uuid = chunk.get("chunk_uuid") or chunk.get("id") or ""
            if chunk_uuid and chunk_id_to_group_ids and chunk_relations:
                group_ids = chunk_id_to_group_ids.get(chunk_uuid, [])
                relevant_relations = [r for r in chunk_relations if r.get("group_id") in group_ids]
                if relevant_relations:
                    lines.append("Graph Relations:")
                    for rel in relevant_relations:
                        for triplet in rel.get("triplets") or []:
                            src = (triplet.get("source") or {}).get("name", "")
                            tgt = (triplet.get("target") or {}).get("name", "")
                            rel_ = triplet.get("relation") or {}
                            pred = rel_.get("canonical_predicate", "")
                            ctx = rel_.get("context", "")
                            line = f"  [{src}] -> {pred} -> [{tgt}]: {ctx}"
                            temporal = rel_.get("temporal_details")
                            if temporal:
                                line += f" [Time: {temporal}]"
                            lines.append(line)

            # Extra context
            extra_ids = chunk.get("extra_context_ids") or []
            if extra_ids and additional_context:
                extras = [additional_context[eid] for eid in extra_ids if eid in additional_context]
                if extras:
                    lines.append("Extra Context:")
                    for extra in extras:
                        extra_source = extra.get("source_title", "")
                        extra_content = extra.get("chunk_content") or extra.get("content", "")
                        lines.append(f"  Related Context ({extra_source}): {extra_content}")

            lines.append("---")
            lines.append("")

    return "\n".join(lines)
