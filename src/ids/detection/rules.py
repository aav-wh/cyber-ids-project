"""
ids.detection.rules
-------------------
Rule-based detection layer supplementing the ML models.

Deterministic signature rules catch well-known attack patterns
(e.g. SYN floods) that the ML model might miss at low confidence.
The RuleEngine is applied after ML inference; a rule match overrides
a BENIGN decision to ATTACK with maximum confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Rule:
    """A single detection rule."""
    name:        str
    description: str
    predicate:   Callable[[dict], bool]   # takes named feature dict, returns bool


# ── Built-in rules ────────────────────────────────────────────────────────────

def _syn_flood(features: dict) -> bool:
    """Flag if SYN count is very high relative to total packets."""
    syn = features.get("SYN Flag Count", 0)
    tot = features.get("Total Fwd Packets", 1)
    return syn > 0 and (syn / max(tot, 1)) > 0.9


def _port_scan(features: dict) -> bool:
    """Flag if Destination Port is in a common scan range and duration is short."""
    port  = features.get("Destination Port", 0)
    dur   = features.get("Flow Duration", 1)
    pkts  = features.get("Total Fwd Packets", 0)
    return dur < 500_000 and pkts <= 2 and port in range(1, 1024)


def _large_outbound(features: dict) -> bool:
    """Flag unusually large outbound transfers (potential exfiltration)."""
    fwd_bytes = features.get("Total Length of Fwd Packets", 0)
    return fwd_bytes > 5_000_000  # > 5 MB forward payload


def _high_packet_rate(features: dict) -> bool:
    """Flag extremely high packet rates (DDoS indicator)."""
    pkt_rate = features.get("Flow Packets/s", 0)
    return pkt_rate > 100_000


BUILT_IN_RULES = [
    Rule("SYN_FLOOD",       "High SYN-to-total-packet ratio",    _syn_flood),
    Rule("PORT_SCAN",       "Short-duration low-port scan",       _port_scan),
    Rule("LARGE_OUTBOUND",  "Large outbound data transfer",       _large_outbound),
    Rule("HIGH_PKT_RATE",   "Extremely high packet rate (DDoS)",  _high_packet_rate),
]


class RuleEngine:
    """
    Apply a set of rules to a named-feature dict and return match results.

    Usage
    -----
        engine = RuleEngine()
        matches = engine.evaluate(feature_dict)
        if matches:
            # override ML decision
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules = rules or list(BUILT_IN_RULES)

    def add_rule(self, rule: Rule) -> None:
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    def evaluate(self, features: dict) -> list[Rule]:
        """
        Evaluate all rules against the feature dict.

        Returns
        -------
        list of matching Rule objects (empty = no match)
        """
        return [r for r in self._rules if r.predicate(features)]

    def evaluate_and_override(
        self,
        features: dict,
        ml_result: dict,
    ) -> dict:
        """
        Apply rules and override the ML result if any rule fires.

        Adds a 'rule_matches' key to the result listing triggered rules.

        Parameters
        ----------
        features  : named feature dict
        ml_result : output of classify_flow()

        Returns
        -------
        Modified ml_result dict
        """
        matches = self.evaluate(features)
        result  = dict(ml_result)

        if matches:
            result["final_decision"] = "ATTACK"
            result["rule_override"]  = True
            result["rule_matches"]   = [r.name for r in matches]
        else:
            result["rule_override"]  = False
            result["rule_matches"]   = []

        return result
