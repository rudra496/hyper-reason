"""
Step-Value Evaluator & Self-Consistency Verifier
Provides real-time rewards, value estimation, and self-correction signals for test-time tree search nodes.
"""

import math
import re
from typing import List, Dict, Tuple, Optional, Any


class StepValueEvaluator:
    """
    Evaluates intermediate reasoning steps using heuristic metrics, pattern verifiers,
    and mathematical consistency checks.
    """
    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold

    def evaluate_step(self, step_text: str, context_prompt: str) -> float:
        """
        Calculates a real-valued reward score in range [0.0, 1.0] for a candidate reasoning step.
        Evaluates structural logic, step transition keywords, mathematical soundness, and repetition penalization.
        """
        score = 0.50  # baseline neutral reward

        # Reward logical transition markers (Reasoning Traces)
        logical_connectives = [
            r"therefore", r"because", r"implies", r"thus", r"consequently",
            r"step \d+", r"let us check", r"hence", r"evaluating", r"substituting"
        ]
        for pattern in logical_connectives:
            if re.search(pattern, step_text, re.IGNORECASE):
                score += 0.05

        # Reward explicit self-correction or verification steps
        self_correct_markers = [
            r"wait,", r"let's re-verify", r"double checking", r"holding back",
            r"correcting calculation", r"sanity check"
        ]
        for pattern in self_correct_markers:
            if re.search(pattern, step_text, re.IGNORECASE):
                score += 0.12

        # Reward numerical equations and structured formulas
        equations = re.findall(r"\d+\s*[\+\-\*\/\=]\s*\d+", step_text)
        if len(equations) > 0:
            score += 0.10

        # Penalize excessive verbosity or endless repetition
        words = step_text.strip().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.40:
                score -= 0.25  # high repetition penalty

        # Penalize refusal or vague output
        if re.search(r"i cannot answer|as an ai|unclear|unknown", step_text, re.IGNORECASE):
            score -= 0.40

        return max(0.0, min(1.0, score))


class SelfConsistencyVerifier:
    """
    Aggregates multi-path rollout outcomes and calculates consensus confidence.
    """
    def __init__(self, agreement_threshold: float = 0.65):
        self.agreement_threshold = agreement_threshold

    def extract_final_answer(self, reasoning_trace: str) -> str:
        """Extracts final boxed or targeted answer string from reasoning trajectory."""
        boxed = re.findall(r"\\boxed\{([^}]+)\}", reasoning_trace)
        if boxed:
            return boxed[-1].strip()
            
        final_line = reasoning_trace.strip().split("\n")[-1]
        numbers = re.findall(r"-?\d+(?:\.\d+)?", final_line)
        if numbers:
            return numbers[-1]

        return final_line[:100].strip()

    def calculate_consensus(self, candidate_outputs: List[str]) -> Tuple[str, float, Dict[str, int]]:
        """
        Computes consensus majority vote across parallel tree rollout leaves.
        Returns: (Best Answer, Confidence Score, Answer Distribution)
        """
        if not candidate_outputs:
            return "", 0.0, {}

        answers = [self.extract_final_answer(out) for out in candidate_outputs]
        counts: Dict[str, int] = {}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1

        best_answer = max(counts, key=counts.get)
        confidence = counts[best_answer] / len(candidate_outputs)

        return best_answer, round(confidence, 4), counts
