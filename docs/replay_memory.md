# Replay Memory for Continual Learning

Replay memory is a simple continual-learning mechanism for reducing catastrophic forgetting.

## Motivation

When a model is trained sequentially on new domains, such as new cities or weather conditions, it may forget previous domains. Replay reduces forgetting by mixing a small number of stored old samples into later training stages.

## Utility

`src/research/replay_memory.py` provides:

- `ReplayItem`
- `ReplayMemory`

## Example

```python
from src.research.replay_memory import ReplayMemory

memory = ReplayMemory(capacity=100, seed=42)
memory.extend(camvid_samples, domain="camvid")

batch = memory.sample(batch_size=8)
```

## Research use

Use replay memory as the first continual-learning baseline before adding stronger methods such as knowledge distillation, adapters, or LoRA.

## Recommended ablations

- memory size: 0, 25, 50, 100, 500,
- replay ratio: 0%, 10%, 25%, 50%,
- replacement policy: random, balanced, uncertainty-based,
- domain sequence: CamVid to Cityscapes to BDD100K to ACDC.
