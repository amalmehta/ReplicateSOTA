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

   → Built the from-scratch model (unit tests pass, first generation coherent) and pushed the public repo. GitHub Actions never started: it sat queued, then failed to start with an internal GitHub error. A local conda-forge PyTorch 2.14 environment replaces it as the reference. A recall pilot found a forget-gate initialisation bug, which is now fixed. Site direction A was built and is waiting for results.
