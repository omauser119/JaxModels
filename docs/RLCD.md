# RLCD: what is public and what Abel implements

TypeSafe introduces **Reinforcement Learning for Calibrated Decisions (RLCD)** as post-training for probabilistic decisions. Its Jev announcement describes a proprietary architecture and parallel sampler with structured numerical outputs instead of generated text. [Official announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev).

TypeSafe's primer contrasts this direction with RLHF and RLVR. Calibration concerns the relationship between predicted probabilities and empirical outcomes across many decisions; a confidence number is not proof that an individual answer is correct. [Official machine-learning primer](https://docs.typesafe.ai/introduction/machine-learning-primer).

The public primitive contract defines Choice, Score and Noul around state plus runtime question definitions, with independent questions. This interface inspires Abel's typed decision API. [Official primitives documentation](https://docs.typesafe.ai/primitives).

## What cannot honestly be specified

These materials do not provide enough detail to reproduce Jev's base model, layer layout, attention scheme, sampler internals, exact RLCD reward, optimizer, training curriculum, training data or weights. We therefore cannot publish a complete authentic Jev/RLCD architecture. Any such recipe here would be an unverified invention. Public interface similarity does not imply shared internals or matching capability.

## What this repository actually provides

Abel has a fully specified frozen backbone plus residual head, soft-target distillation and temperature fitting. [ARCHITECTURE.md](ARCHITECTURE.md) gives its equations, optimizer, data flow and code map. No reinforcement-learning algorithm is implemented in the current trainer. Neither temperature scaling nor cross-entropy distillation is labeled RLCD.

A future RL research implementation would need an explicit policy and action space, a proper-scoring reward/objective, a documented update algorithm and independently held-out calibration and task evaluations. That would be a new, separately versioned implementation, with its own evidence; it would not establish reproduction of TypeSafe's undisclosed training method. There is no claim of AGI, superhuman quality or frontier equivalence in this release.
