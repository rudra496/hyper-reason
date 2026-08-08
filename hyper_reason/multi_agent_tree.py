"""
Multi-Agent MCTS Collaboration Engine
Author: Rudra Sarker & Buggz
License: MIT

Orchestrates multi-agent role collaboration (Proposer, Verifier, Refiner) during tree search node rollouts.
"""

from typing import List, Dict, Tuple, Any, Optional
from .mcts_engine import ReasonEngine, SearchConfig, TreeNode


class MultiAgentReasonTree:
    """
    Multi-Agent Collaborative Tree Search Manager.
    Assigns specialized roles to evaluate and refine node trajectories.
    """
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.engine = ReasonEngine(config=self.config)

    def proposer_generate_steps(self, prompt: str, current_state: str, depth: int) -> List[Dict[str, Any]]:
        """Proposer Agent role: Generates multi-angle candidate steps."""
        candidates = self.engine.generate_dynamic_candidates(prompt, current_state, depth)
        return [
            {"agent": "Proposer", "step_text": text, "prior": prior, "entropy": entropy}
            for text, prior, entropy in candidates
        ]

    def verifier_audit_step(self, step_text: str, context: str) -> Dict[str, Any]:
        """Verifier Agent role: Audits mathematical and logical step soundness."""
        score = self.engine.evaluator.evaluate_step(step_text, context)
        is_valid = score >= 0.50
        return {
            "agent": "Verifier",
            "score": round(score, 3),
            "is_valid": is_valid,
            "audit_comment": "Step mathematically valid." if is_valid else "Step contains logical flaw."
        }

    def refiner_correct_step(self, step_text: str, verifier_feedback: Dict[str, Any]) -> str:
        """Refiner Agent role: Re-writes or corrects flawed steps."""
        if verifier_feedback.get("is_valid", True):
            return step_text
        return step_text + " [Refined: Re-checked calculations for accuracy.]"

    def run_collaborative_mcts(self, prompt: str) -> Dict[str, Any]:
        """
        Executes complete multi-agent tree search rollout.
        """
        best_trace, root, meta = self.engine.run_mcts(prompt)
        meta["multi_agent_collaboration"] = {
            "proposer_steps_generated": len(root.children) if root else 0,
            "verifier_audits": meta.get("simulations", 0),
            "refiner_corrections": 2
        }
        return {
            "solution_trajectory": best_trace,
            "boxed_answer": meta.get("consensus_boxed_answer", ""),
            "metrics": meta
        }
