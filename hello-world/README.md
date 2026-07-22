# Hello World

## Description

Train a linear regression model on a simple 2D dataset (x → y with approximately linear relationship) and produce predictions on a held-out test set. This is a minimal ML-Bench example task designed to demonstrate the task format and test infrastructure.

## Capability Tested

Tool fluency — basic ML pipeline: read data, fit model, write predictions.

## Scoring

Binary. All of: predictions in plausible range, monotonically increasing, within tolerance of expected values.

## Completion Rates

| Model | Pass | Median wall-clock (s) | Failure modes |
|---|---|---|---|
| Oracle | 3/3 | — | n/a |

## Model Analysis

This is a hello-world example. All models should pass easily.

## Anti-Cheating Analysis

- Range check [30, 60] prevents hardcoded out-of-range values
- Monotonicity check prevents random or constant predictions
- Tolerance check (±2.0) against expected values catches wrong algorithms
- Baseline floor requires beating mean prediction
- Holdout check verifies generalization to unseen x values

## Reproducibility

Deterministic. Seed = 42. Linear regression has a closed-form solution — no stochastic variance.
