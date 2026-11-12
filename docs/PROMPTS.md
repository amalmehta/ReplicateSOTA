# Prompt Log

Every prompt given to the agent, word for word, in order, with a line on what it led to.

## 2026-11-11 — Kick-off: read the spec and start the build

1. > replicate any SOTA model: build this project. Read replicate any SOTA model.md in this folder and build what it describes, following its META-INSTRUCTIONS exactly.

   → Read the spec and Shortcuts.md, checked this Mac (Intel i9, 32 GB RAM, no GPU, 40 GB disk free), and asked the first round of questions on purpose, scope and form before building anything.

   Answers given in the question widget, in order (summarised; options chosen):
   - Goals: all four picked (understand, teaching site, pipeline, research), then ranked: **understand deeply** wins conflicts.
   - "Replicate" means: all four picked, then v1 narrowed to **logit match + tiny training**. Reason: "Matching = copied math" (training shows understanding).
   - First model: **Qwen3.5-0.8B**. Form: **repo + static explainer site**. Repo: **public, $0, CPU only**.
   - Experiment: **MQAR recall test**. Reference: **GitHub Actions**. Reader: **knows transformer basics**.
   - Done criteria: **accepted as written**. Push timing: **push early, README marked WIP**.
   - CI blocked: **user checks GitHub billing; local fallback meanwhile**. Site look: **A, editorial long-read**.

   → Built the from-scratch model (unit tests pass, first generation coherent) and pushed the public repo. GitHub Actions never started: it sat queued, then failed to start with an internal GitHub error. A local conda-forge environment (PyTorch 2.7.1, transformers 5.22.2) replaces it as the reference. A recall pilot found a forget-gate initialisation bug, which is now fixed. Site direction A was built and is waiting for results.

## 2026-11-11 — Checks, threshold and results

2. > what do i need to check?

   → Listed what to check by hand: the logit-match table, the site in light and dark mode and at phone width, and the recall setup against the papers.

3. > tighten the threshold to 1e-4

   → Changed it from 1e-3 to 1e-4 in `reference/make_reference.py`, `tests/test_logit_match.py`, the README, the site and the spec. The worst measured gap is 5.3×10⁻⁵, so it still passes.

4. > push it when the tests pass

   → All 8 logit-match tests passed at 1e-4 (23 min 42 s). The change was already on GitHub as commit d9aef38, committed from outside this session before the tests finished, so no second commit was made.

5. > show me the recall results when they're done

   → All 18 runs finished. At N=64 the three layer types tie (97.8 / 97.9 / 100%). At N=128 only 1 of 3 softmax seeds learned the task, so its averages measure training speed, not memory. Raised this and offered two options: mark it inconclusive, or rerun with more steps.

6. > Per-seed dots, inconclusive

   → The chart now shows every seed as a dot, with the mean as a bar. N=128 is shaded and labelled inconclusive. The verdict is given per length: N=64 is a tie (−0.2 points), N=128 is inconclusive. The ± values now use sample standard deviation. The new rule, that a length counts only if softmax reaches ≥ 90% on every seed, is marked on the site as added after seeing the results. No reason was given for the choice.
