import json
import re

from peek.core.types import Usage


class FakeSemanticPeekClient:
    """Deterministic test double; intentionally lives outside production package."""
    def __init__(self): self.candidate = ""; self.calls = 0
    def completion(self, messages):
        self.calls += 1; text = messages[-1]["content"]
        if "trace" in text.lower() or "trajectory" in text.lower():
            matches = re.findall(r'"text":\s*"([^"]+)"', text)
            evidence = matches[0].replace("\\n", " ") if matches else "missing evidence"
            self.candidate = f"Corpus evidence says: {evidence}"
            return json.dumps({"diagnosis": "grounded semantic fact", "item_tags": {},
                               "cache_candidates": [{"content": self.candidate}]})
        return json.dumps({"reasoning": "store readable evidence", "operations": [
            {"type": "ADD", "section": "context_understanding", "content": self.candidate}]})
    def last_usage(self): return Usage(11, 7)
