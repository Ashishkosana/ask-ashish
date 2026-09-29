# review-lens

Portfolio site card: Self-verifying LLM code reviewer — runs independent lenses (correctness, security, performance, tests), then adversarially checks its own findings to cut false positives. Stack: Python · LLM · GitHub Action. Published site line: evaluation harness · precision/recall/F1 · 81 tests · mypy --strict.

From the public README: review-lens reviews a git diff across four lenses — correctness, security, performance, and test-coverage — then runs an adversarial self-verification pass that tries to refute each finding before you see it. It suggests. You decide. Nothing is ever auto-applied. The README says the evaluation harness measures precision, recall, and F1, and that those numbers are not hardcoded in the README. This note does not add a score the README does not print.

Source: https://github.com/Ashishkosana/review-lens
