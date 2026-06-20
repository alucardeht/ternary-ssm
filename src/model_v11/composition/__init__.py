"""Model composition and merging techniques.

This module implements methods for merging multiple model instances,
including TIES merging for specialized model combination and growth strategies.
"""

from .ties_merge import TernaryTIESMerger, validate_merge_quality
from .growth import (
    Net2NetGrower,
    BranchTrainMerge,
    MultiExpertVoter,
    LotteryTicketFinder,
    BayesianLRReset,
)

__all__ = [
    "TernaryTIESMerger",
    "validate_merge_quality",
    "Net2NetGrower",
    "BranchTrainMerge",
    "MultiExpertVoter",
    "LotteryTicketFinder",
    "BayesianLRReset",
]
