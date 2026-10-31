# Symbolic Core and Corpus (Tensegrity 2.3.0, "Symbolic Genesis" – sub-project 1)

## 1. Purpose and scope

Give Tensegrity a formally defined symbolic language, a deterministic runtime for it, and a reproducible
synthetic corpus, so a later cycle can train a model to generate typed symbolic agent definitions and
measure the result by execution rather than by appearance.

**In scope (this cycle):** the language (types, grammar, parser, type checker, interpreter, canonical
serializer), the synthetic corpus generator (curriculum stages A-E), the leakage auditor, a CLI entry point,
and tests.

**Out of scope (later sub-projects):**
- Sub-project 2: wiring into training (mixture and curriculum scheduling), model-facing evaluation metrics,
  grammar-constrained decoding, a trained 1M baseline, recording the corpus manifest hash in run metadata.
- Sub-project 3: agent lifecycle (conceive/instantiate/evaluate/terminate), dependency graphs of agents,
  curriculum stages F-G, a desugaring/lowering pass or binary encoding if ever needed.

**Non-goals / invariants:** no change to `ByteTokenizer`, `TensegrityConfig`, model architecture, checkpoint
format or growth ladder. The `symbolic` package must not import torch.

**Versioning:** the milestone is named "Symbolic Genesis" and ships as package version 2.3.0 (the repo is
already at 2.2.0; the "1.0" label from the original proposal is not used).

## 2. Language

### 2.1 Types
`Int`, `Bool`, `List[T]`, and function types `T -> U`. Ints are Python ints bounded in magnitude by a
configurable `int_bound`; exceeding it is a deterministic `EvalError` (no wraparound).

### 2.2 Syntax (S-expressions)
```lisp
(lambda ((x Int)) (add x 1))
(let ((y (mul 2 3))) (add y 1))
(if (lt x 0) (neg x) x)
(compose (add 2) (mul 3))          ; (compose f g) x = f(g(x))
(map (add 1) (list 1 2 3))
(filter (lt 2) (list 1 2 3))
(fold add 0 (list 1 2 3))          ; fold f init xs, left fold
(apply f x)                        ; (f x) application is also accepted
(agent (input (x Int)) (output Int) (body (add x 1)))
```
Primitives: `add sub mul neg`, `eq lt le`, `and or not`, `list cons nil head tail len`.
Binary primitives support partial application (`(add 2) : Int -> Int`). `head`/`tail` of an empty list raise
`EvalError`.

### 2.3 Canonical form
Single spaces, no comments, fixed field order, lambda parameters always typed. One AST has exactly one
serialization, and `parse(serialize(ast)) == ast`. The canonical bytes are UTF-8 and ASCII-only.

### 2.4 Agent form
`(agent (input ...) (output T) (body e))` is a closed, typed, single-use computation. This cycle only requires
it to parse, type-check and evaluate for a given input. It appears in stage C documents on purpose
(documented curriculum overlap with the later stage F).

### 2.5 Guarantees
- No recursion or general loops: evaluation terminates. A step budget (fuel) still bounds work; exhaustion is
  a deterministic `EvalError`.
- No I/O, clock or randomness in the interpreter.
- Typed errors: `ParseError`, `TypeError_`, `EvalError`. The checker returns a stable reason code for each
  failure (used for negative-example labels).

## 3. Package layout

```
tensegrity/symbolic/
    __init__.py
    algebra.py        # types, AST nodes, primitive signatures
    grammar.py        # tokenizer/parser, canonical serializer
    typecheck.py      # bidirectional checker, reason codes
    interpreter.py    # pure tree-walking evaluator with fuel and int bound
    prng.py           # self-contained splitmix64 PRNG + per-document seeding
    sampler.py        # type-directed AST sampler + productivity (feasibility) table
    corpus.py         # curriculum stages A-E, mutation pass, splitting, writer, manifest, resume
    verification.py   # audit_corpus(): independent leakage / label / dedup / EOS checks
    config.py         # SymbolicConfig
```
Added to the package's lazy `_EXPORTS` table where useful. New CLI subcommand:
`tensegrity symbolic-corpus --seed N --out DIR [--config FILE]`.

`SymbolicConfig` (dataclass, validated, serializable): `max_depth`, `enabled_ops`, `stage_weights`,
`n_train`, `n_val`, `n_test_comp`, `held_out_pairs`, `error_policy` (`keep_labelled` | `reject`), `int_bound`,
`fuel`, `shard_size`, `seed`. It is independent of `TensegrityConfig`.

## 4. Corpus generation

### 4.1 Determinism
- Each candidate gets its own RNG seeded from
  `sha256(f"tensegrity:v1:{seed}:{split}:{candidate_index}")`, driving a splitmix64 PRNG implemented in
  `prng.py`. Python's built-in `hash()` and `random.Random` are not used (their behaviour is not stable across
  processes / CPython versions).
- Each record stores `candidate_index` (immutable identity of the attempt) and `accepted_index` (position in
  the output). Rejected candidates keep their index; `accepted_index` is assigned in candidate order.
- Candidates are generated in fixed-size shards. Resuming regenerates only the incomplete shard, so an
  interrupted-and-resumed run is byte-identical to an uninterrupted one.

### 4.2 Sampling
A type-directed sampler builds a well-typed AST for a requested type, depth and operator set; it never
generates and filters. A productivity table (per type and depth) is computed from the enabled operators
first; an infeasible request raises `ConfigError` immediately, so sampling always terminates. The yield
guarantee is: every feasible (type, depth, operator-set) request produces a well-typed AST. Well-typed
programs can still fail evaluation (`head`/`tail` of empty, int bound, fuel); `error_policy` decides whether
those become labelled error examples or are rejected.

### 4.3 Stages
Each document is plain UTF-8 text of at most a few hundred bytes.

| Stage | Task | Document |
|---|---|---|
| A | Validity | `CHECK <program>` -> `OK <type>` or `ERR <kind>: <reason>` |
| B | Completion | `COMPLETE <program with one hole _>` -> `<full program>` |
| C | Evaluation | `EVAL <program> @ <input>` -> `<value>` (agent forms allowed) |
| D | Synthesis | `SYNTH (<in> <out>) x k` -> `<program>` |
| E | Composition | `COMPOSE <f> <g>` -> `<canonical (compose f g)>` and its `EVAL` |

Stage A negatives come from a mutation pass over valid programs (dropped paren, unknown operator, arity
error, type mismatch, unbound variable). Every mutant is run through the real checker; its label is whatever
the checker reports (a mutant that remains valid is labelled `OK`).

### 4.4 Splits and compositional hold-out
- A *pair* is an ordered parent->child operator edge in an AST (e.g. `fold` with a `compose` function
  argument). `held_out_pairs` are combinations of operators that each appear elsewhere in training; `test_comp`
  tests composition, not new vocabulary.
- Assignment order: (1) structural eligibility: if **any** AST appearing in the document (program, hole
  completion, COMPOSE operands/result, synth target) contains a held-out pair, the document goes to
  `test_comp`; (2) otherwise `sha256(canonical_program, seed) mod 10000` against the train threshold selects
  `train` or `val`.
- A canonical program lives in exactly one split. Deduplication is on the full serialized document, within
  each split.
- `test_comp` is an evaluation artifact and is never part of a training mixture.

### 4.5 Evaluation contract (recorded in the records; scoring code lands in sub-project 2)
- Stage D is scored **functionally**: a generated program is correct if it matches all k examples and also
  matches the reference on held-out inputs (behavioral check). Exact match to the reference program is not
  used, since examples can underdetermine the program.
- Structural match (the output contains the intended pair) is reported separately. `test_comp` synth tasks
  carry a `structure_required` flag, restricting the task so a simpler equivalent cannot bypass the pair.

## 5. Output contract

```
DIR/
  train.jsonl  val.jsonl  test_comp.jsonl
  manifest.json
```
One JSON record per document, one document per line. The `text` field is the training document and **contains
no EOS**; `prepare_data` appends exactly one EOS per record. The invariant is one document boundary per
document. Metadata fields: `id`, `stage`, `split`, `candidate_index`, `accepted_index`, `programs`
(canonical programs in the document), `canonical_hash`, `result_type`, `expected`, `structure_required`,
`difficulty = {depth, nodes, pairs, steps}`.

`manifest.json`: seed, full `SymbolicConfig`, language/schema version, per-split and per-stage counts, a
content hash per file and a combined corpus hash, and the held-out pair list.

## 6. Verification

`audit_corpus(dir)` runs independently of the generator and re-parses the finished documents (it does not
trust record metadata). It checks: no held-out pair in `train`/`val`; zero canonical-program overlap across
splits; every `OK`/`ERR` label and every `expected` value matches a fresh checker / interpreter run; no EOS
bytes inside `text`; stage quotas; document-level dedup. It is called at the end of generation and exposed in
the CLI.

## 7. Acceptance criteria (all automated tests)

| Area | Requirement |
|---|---|
| Generation | 100% well-typed for feasible configs; `ConfigError` for infeasible |
| Language | parse/serialize identity on canonical form; interpreter deterministic and terminating within fuel |
| Properties | seeded-loop tests over a few thousand random ASTs: round-trip, type preservation, checker/interpreter agreement (no Hypothesis dependency) |
| Reproducibility | same seed -> byte-identical corpus and manifest; interrupted-then-resumed run identical to uninterrupted |
| Splits | zero canonical overlap across splits; zero held-out pairs in `train`/`val`; `audit_corpus` passes |
| Labels | 100% interpreter-derived `expected` and checker-derived negative labels match re-run |
| Quotas | stage mix matches `stage_weights` |
| Ingestion | `prepare_data` on `train.jsonl` yields exactly one EOS per document; existing suite still passes |
| Throughput | generation benchmark reported, not gated |

## 8. Risks and notes

- A rich hold-out may starve `test_comp` of enough examples at small depth; the generator reports per-split
  counts and fails loudly if a requested split cannot be filled.
- Difficulty metadata is recorded now so later cycles can slice results by depth/size/composition.
- Adding lowering or a binary form later must not change the canonical text form defined here.
