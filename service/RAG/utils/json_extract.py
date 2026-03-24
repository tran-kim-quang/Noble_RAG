from typing import Optional


def extract_first_json_object(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from LLM output text."""
    if not text:
        return None

    def _extract_balanced(source: str) -> Optional[str]:
        start = source.find("{")
        if start < 0:
            return None
        depth = 0
        for idx in range(start, len(source)):
            ch = source[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return source[start: idx + 1]
        return None

    clean = text.strip()

    if "```" in clean:
        for block in clean.split("```"):
            candidate = block.strip()
            if not candidate:
                continue
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            result = _extract_balanced(candidate)
            if result:
                return result

    return _extract_balanced(clean)
