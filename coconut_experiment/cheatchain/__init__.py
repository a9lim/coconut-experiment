"""CheatChain: a synthetic reward-hackable reasoning task.

Variable chains under modular arithmetic, with an optional leaked-answer cue.
Designed for a matched CoT-vs-Coconut study of emergent reward hacking, with
per-example behavioral ground truth via the cue-flip counterfactual.

See the repository-root docs/design.md (§CheatChain) for the design.
"""

from .generate import (
    Condition,
    Step,
    Chain,
    CheatChainExample,
    GenConfig,
    gen_chain,
    build_example,
    make_training_set,
    make_eval_set,
    flipped,
    verify_example,
    dataset_stats,
)

__all__ = [
    "Condition",
    "Step",
    "Chain",
    "CheatChainExample",
    "GenConfig",
    "gen_chain",
    "build_example",
    "make_training_set",
    "make_eval_set",
    "flipped",
    "verify_example",
    "dataset_stats",
]
