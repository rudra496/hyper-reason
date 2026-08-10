"""The honest HyperReason engine core.

Pure Python (stdlib + numpy at most). No torch/requests/httpx here — model access goes
through ``hyper_reason.backends``. This keeps the core portable (and JS-portable for the
website) and makes every number traceable to a real backend call.
"""

from .config import SearchConfig
from .math_utils import (
    shannon,
    shannon_from_counts,
    compute_salience,
    extract_boxed,
    extract_final_answer,
    normalize_step,
)

__all__ = [
    "SearchConfig",
    "shannon",
    "shannon_from_counts",
    "compute_salience",
    "extract_boxed",
    "extract_final_answer",
    "normalize_step",
]
