"""
Persistent Reasoning Memory Store Module
Author: Rudra Sarker & Buggz
License: MIT

Stores and retrieves verified high-reward reasoning trajectories across multi-turn sessions.
"""

from typing import Dict, List, Optional, Any
import json
import re


class ReasoningMemoryStore:
    """
    Lightweight key-pattern memory store for reasoning trajectories.
    """
    def __init__(self):
        self.memory: Dict[str, Dict[str, Any]] = {}

    def _extract_keywords(self, prompt: str) -> str:
        """Extracts normalized key pattern signature from prompt text."""
        words = re.findall(r"\w+", prompt.lower())
        return "_".join(sorted(set(words[:8])))

    def store_trajectory(self, prompt: str, solution_trajectory: str, boxed_answer: str, confidence: float):
        """Stores a high-confidence reasoning trajectory into memory."""
        key = self._extract_keywords(prompt)
        self.memory[key] = {
            "prompt": prompt,
            "solution_trajectory": solution_trajectory,
            "boxed_answer": boxed_answer,
            "confidence": confidence
        }

    def recall_trajectory(self, prompt: str) -> Optional[Dict[str, Any]]:
        """Recalls matching stored reasoning trajectory if pattern exists."""
        key = self._extract_keywords(prompt)
        return self.memory.get(key)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Returns aggregate memory store statistics."""
        return {
            "total_remembered_trajectories": len(self.memory),
            "memory_keys": list(self.memory.keys())
        }
