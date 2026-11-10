# Claude Development Instructions

## Development workflow

Claude is the sole development assistant for this repository. Complete planning, implementation, review, and verification in the current Claude session. Do not require a Codex review, a second assistant's approval, or a Claude/Codex handoff loop to finish a task.

Start with [the development guide](docs/DEVELOPMENT.md) for the repository map, Windows commands, and validation limits. Read the current implementation status before making claims about completed features. Keep each change scoped to the user's request, review the diff, and report the checks actually run and any remaining limitations.

## Authoritative specification

Before planning or changing application behavior, read [the design specification](docs/design-specification.md). It is the baseline for future development and takes precedence over descriptions of intended behavior in README files and development or setup guides.

The specification describes the target application. Existing code and historical implementation notes are not evidence that a requirement is complete. Keep current implementation status distinct from planned behavior. Resolve conflicts within the scope of the requested task; do not silently rewrite the specification or claim unmeasured accuracy.

Follow the six development phases in Section 35 when prioritizing new work. Do not implement unrelated phases merely because they appear in the specification. This documentation baseline does not itself authorize application changes.

## Recognition and grouping invariants

- Minimize merges of different people; accuracy takes priority over speed.
- Use multiple high-quality face samples and embeddings, not a single frame or maximum similarity alone.
- Use JAPANESE FACE V1 as the primary model and AdaFace as a separate auxiliary model when its phase is implemented.
- Keep model embedding spaces and model versions separate. Track model-specific preprocessing.
- Only HIGH decisions are eligible for automatic merging. MEDIUM decisions require human review. Model disagreement must not trigger an automatic merge.
- Do not merge clusters through transitive similarity alone. Compare a proposed addition against multiple tracks in the cluster.
- Preserve user-confirmed same-person and different-person decisions, and enforce negative-pair constraints during automatic merging.
- Support local Windows execution, CPU fallback, caching, incremental analysis, and replaceable model adapters.
- Treat example settings and historical thresholds as uncalibrated until evaluated on representative videos.

## Evidence

- Be candid and logical. Identify incorrect claims rather than agreeing for convenience. State when something is unknown; never invent results.
- Seek current information when the task needs it. Verify changing technical facts using primary sources, and report dates or versions where relevant.
- Prefer quantitative results when supported. Distinguish measurements, estimates, examples, and unverified assumptions.
- Do not report tests, launches, benchmarks, or recognition accuracy as verified unless they were actually checked.

## Language and documentation

- Implement software and application UI in English by default.
- Use English for code and general documentation intended for GitHub. User-only material may be Japanese.
- Write README content for first-time visitors and non-engineers, emphasizing what the application can do and keeping technical detail limited.
- Keep README documentation in English only; do not add translated README files or language navigation.
- The user's original Japanese design is kept locally in `docs/private/design-specification.ja.md`. It is excluded from Git; the public development specification is `docs/design-specification.md`.
