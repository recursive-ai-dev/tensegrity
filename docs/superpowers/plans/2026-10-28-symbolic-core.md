# Symbolic Core and Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tensegrity.symbolic`: a typed S-expression language with parser, type checker, pure interpreter and canonical serializer, plus a deterministic, resumable synthetic corpus generator (stages A-E) with a leakage auditor and a `tensegrity symbolic-corpus` CLI.

**Architecture:** One typed AST is the IR. `parse -> typecheck -> evaluate`, with `serialize` giving the unique canonical text. Corpus shards are pure functions of `(seed, config, candidate range)`; assembly replays shards in order to deduplicate, enforce cross-split disjointness and fill quotas. Output is JSONL consumable by the existing `prepare_data`.

**Tech Stack:** Python >=3.9 stdlib only inside `tensegrity/symbolic/` (no torch, no numpy). pytest. Run tests with `.venv/bin/python -m pytest`.

**Spec:** `docs/superpowers/specs/2026-10-28-symbolic-core-design.md`

## Global Constraints

- `tensegrity/symbolic/` imports nothing from torch/numpy; only `tensegrity.symbolic.*` and stdlib. (Spec §1)
- No change to `ByteTokenizer`, `TensegrityConfig`, model, checkpoint format, growth ladder. (Spec §1)
- Package version becomes `2.3.0` in `pyproject.toml` and `tensegrity/__init__.py`; milestone name "Symbolic Genesis". (Spec §1)
- Canonical text is ASCII-only, single-spaced, no comments; `parse(serialize(a)) == a`. (Spec §2.3)
- `text` fields in corpus JSONL contain **no EOS**; `prepare_data` appends exactly one per record. (Spec §5)
- Per-candidate RNG seeded from `sha256(f"tensegrity:v1:{seed}:{split}:{candidate_index}")`; no `hash()`, no `random.Random`. (Spec §4.1)
- Python >=3.9 compatible syntax (no `match`, no `X | Y` type unions at runtime).
- Known pre-existing failure, do not fix or touch: `tests/test_notebook.py` (looks for `notebooks/Tensegrity_Colab.ipynb`, repo has `..._0.9.0.ipynb`). The notebook is not modified by this plan.

## Decisions the spec left open (fixed here)

- Type syntax in text: `Int`, `Bool`, `(List T)`, `(-> A B)` (curried, so `(-> Int Int Int)` is `Int -> Int -> Int` sugar for `(-> Int (-> Int Int))`; serialize always emits the nested form).
- Canonical application is `(f a b)`; `(apply f a)` is accepted by the parser and normalised to `(f a)`.
- Empty list is `(nil T)`; `(list)` with no items is a parse error.
- Literals: `-?[0-9]{1,12}` and `true`/`false`. Hole `_` parses only with `parse(..., allow_holes=True)`.
- The `{split}` slot in the RNG seed holds the fixed label `"pool"`: a document's split depends on its content, so candidates come from one pool stream and are classified afterwards (refinement of spec §4.1).
- Sampler type universe: `Int`, `Bool`, `(List Int)`, `(List Bool)`, `(-> Int Int)`, `(-> Int Bool)`, `(-> Bool Bool)`, `(-> Int Int Int)`, `(-> (List Int) Int)`, `(-> (List Int) (List Int))`.
- Default `held_out_pairs`: `(("map","compose"), ("filter","compose"), ("fold","lambda"))`.
- Extra `SymbolicConfig` fields beyond the spec list: `synth_examples=4`, `synth_holdout=4`, `pair_focus_prob=0.15`, `max_candidates` (default `50 * (n_train+n_val+n_test_comp)`).

## Review Focus

1. Pathological text to `parse` (10 000 nested parens, NUL bytes, non-ASCII, empty string, 5 000-digit literal) -> `ParseError`, never `RecursionError`/hang. Owned by Task 2.
2. Resuming into an output dir whose manifest/shard headers have a different seed or config -> refuse with a clear error, never mix shards. Owned by Task 9.
3. Crash residue: stray `*.tmp` file or a shard with a truncated JSON line -> regenerated, not trusted. Owned by Task 9.
4. A quota that cannot be filled (e.g. `n_test_comp` too large for the hold-out pairs, or `max_candidates` exhausted) -> loud `CorpusError` naming the split and counts. Owned by Task 9.
5. Multiplication chains overflowing `int_bound` and deep `fold` over long lists hitting `fuel` -> deterministic `EvalError`, labelled (or rejected per `error_policy`), never a Python exception. Owned by Tasks 4 and 8.

## File Structure

```
pyproject.toml                         # packages += tensegrity.symbolic; version 2.3.0
tensegrity/__init__.py                 # __version__ = "2.3.0"
tensegrity/cli.py                      # + symbolic-corpus subcommand
tensegrity/symbolic/__init__.py        # public re-exports
tensegrity/symbolic/errors.py          # ParseError, TypeError_, EvalError, ConfigError, CorpusError
tensegrity/symbolic/algebra.py         # types, AST nodes, primitive table
tensegrity/symbolic/grammar.py         # parse, serialize, type_to_str
tensegrity/symbolic/typecheck.py       # infer / check
tensegrity/symbolic/interpreter.py     # evaluate, render_value
tensegrity/symbolic/config.py          # SymbolicConfig
tensegrity/symbolic/prng.py            # SplitMix64, document_rng
tensegrity/symbolic/sampler.py         # productivity table, sample_expr, sample_value
tensegrity/symbolic/splits.py          # op_name, collect_pairs, assign_split
tensegrity/symbolic/stages.py          # stage A-E document builders, mutation pass
tensegrity/symbolic/corpus.py          # shards, assembly, writer, manifest, resume
tensegrity/symbolic/verification.py    # audit_corpus
tests/test_symbolic_language.py        # Tasks 1-4
tests/test_symbolic_generation.py      # Tasks 5-8
tests/test_symbolic_corpus.py          # Tasks 9-11
```
(`splits.py` and `stages.py` are split out of the spec's single `corpus.py` to keep each file focused; spec's `conception/lifecycle/compiler` remain deferred.)

---

### Task 1: Scaffold, errors, algebra

**Files:**
- Create: `tensegrity/symbolic/__init__.py`, `errors.py`, `algebra.py`
- Modify: `pyproject.toml:22-23` (`packages = ["tensegrity", "tensegrity.symbolic"]`)
- Test: `tests/test_symbolic_language.py`

**Interfaces:**
- Produces (`errors.py`): `class SymbolicError(Exception)`; subclasses `ParseError(code: str, detail: str = "")`, `TypeError_(code: str, detail: str = "")`, `EvalError(code: str, detail: str = "")`, `ConfigError`, `CorpusError`. The first three expose `.code` and `.kind` (`"parse"`, `"type"`, `"eval"`).
- Produces (`algebra.py`): frozen dataclasses `IntT`, `BoolT`, `ListT(elem)`, `FunT(arg, ret)` (type alias `Type`); expression nodes `Lit(value: Union[int,bool])`, `Var(name)`, `Prim(name)`, `App(fn, args: Tuple[Expr,...])`, `Lambda(params: Tuple[Tuple[str,Type],...], body)`, `Let(bindings: Tuple[Tuple[str,Expr],...], body)`, `If(cond, then, other)`, `Compose(f, g)`, `MapE(f, xs)`, `FilterE(f, xs)`, `FoldE(f, init, xs)`, `ListLit(items: Tuple[Expr,...])`, `Nil(elem: Type)`, `Hole()`, `Agent(params, out: Type, body)`. `PRIMS: Dict[str, Tuple[int, ...]]` names with arity (`add sub mul eq lt le and or cons`: 2; `neg not head tail len`: 1). Helper `children(e: Expr) -> Tuple[Expr, ...]` (direct sub-expressions in left-to-right order) and `size(e) -> int`, `depth(e) -> int`.

- [ ] **Step 1: Write failing tests** — `test_errors_carry_kind_and_code` (`ParseError("x").kind == "parse"`), `test_nodes_are_hashable_and_equal_by_value` (`App(Prim("add"),(Lit(1),)) == App(Prim("add"),(Lit(1),))` and usable in a `set`), `test_children_size_depth` (`size(App(Prim("add"),(Lit(1),Lit(2)))) == 4`, `depth(Lit(1)) == 1`), `test_pyproject_lists_symbolic_package` (reads `pyproject.toml` text, asserts `"tensegrity.symbolic"` present).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_symbolic_language.py -q` -> FAIL (module missing).
- [ ] **Step 3: Implement** the modules above with the exact names; edit `pyproject.toml`.
- [ ] **Step 4: Run** same command -> PASS.
- [ ] **Step 5: Commit** `git add tensegrity/symbolic tests/test_symbolic_language.py pyproject.toml && git commit -m "feat(symbolic): scaffold errors and typed AST"` (end message with the attribution line from the session reminder).

---

### Task 2: Grammar (parser and canonical serializer)

**Files:** Create `tensegrity/symbolic/grammar.py`; Test `tests/test_symbolic_language.py`

**Interfaces:**
- Consumes: Task 1 nodes/errors.
- Produces: `parse(text: str, *, allow_holes: bool = False, max_depth: int = 64) -> Expr`; `serialize(e: Expr) -> str`; `type_to_str(t: Type) -> str`; `parse_type(text: str) -> Type`. `ParseError` codes: `empty`, `unbalanced`, `trailing`, `unexpected_token`, `bad_literal`, `bad_type`, `too_deep`, `non_ascii`, `unknown_form` (wrong shape for a special form, e.g. `(if 1 2)`).

- [ ] **Step 1: Write failing tests**: `test_roundtrip_spec_examples` (each of the spec §2.2 snippets: `serialize(parse(s)) == s`); `test_apply_alias_is_normalised` (`serialize(parse("(apply f x)")) == "(f x)"`); `test_canonical_whitespace` (`serialize(parse("(add   1\n 2)")) == "(add 1 2)"`); `test_nil_and_empty_list_rule` (`parse("(nil Int)")` ok, `parse("(list)")` -> `ParseError` code `unknown_form`); `test_type_syntax_curried` (`type_to_str(parse_type("(-> Int Int Int)")) == "(-> Int (-> Int Int))"`); `test_hole_requires_flag`; **Review Focus 1**: `test_pathological_inputs_raise_parseerror` parametrized over `"(" * 10000`, `"\x00"`, `"(add 1 é)"`, `""`, `"(add " + "9"*5000 + " 1)"` each asserting `pytest.raises(ParseError)` and nothing else; `test_property_roundtrip` (Task 6 sampler is not available yet, so build random ASTs with a small local generator in the test using `random.Random(0)`, 2000 iterations: `parse(serialize(a)) == a`).
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** a hand-written tokenizer + recursive-descent parser with an explicit depth counter (no regex-recursion); `serialize` is a recursive pure function. Reject non-ASCII before tokenizing.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): parser and canonical serializer`.

---

### Task 3: Type checker

**Files:** Create `tensegrity/symbolic/typecheck.py`; Test `tests/test_symbolic_language.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `infer(e: Expr, env: Optional[Dict[str, Type]] = None) -> Type` (raises `TypeError_`), `check(e: Expr, t: Type, env=None) -> None`, `check_text(text: str) -> Tuple[str, str]` returning `("OK", type_str)` or `("ERR", "parse:<code>")` / `("ERR", "type:<code>")` (never raises for bad input). `TypeError_` codes: `unbound_name`, `arity`, `mismatch`, `not_function`, `hole`, `empty_list`. Agent infers to `(-> in... out)` after checking `body : out`.

- [ ] **Step 1: Write failing tests**: `test_infers_spec_examples` (`(compose (add 2) (mul 3))` -> `(-> Int Int)`; `(map (add 1) (list 1 2 3))` -> `(List Int)`; `(fold add 0 (list 1 2 3))` -> `Int`; `(add 2)` -> `(-> Int Int)`); `test_agent_type`; `test_error_codes` (table: `(add 1 true)`->`type:mismatch`, `(add 1 2 3)`->`type:arity`, `(foo 1)`->`type:unbound_name`, `(1 2)`->`type:not_function`, `(add (` -> `parse:unbalanced`); `test_cons_head_polymorphism` (`(cons true (nil Bool))` ok, `(head (list 1))` -> `Int`); `test_hole_is_a_type_error`.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** synthesis-only inference (lambda params are annotated); `cons/head/tail/len/nil` handled as polymorphic special cases; over-application and under-application of `Prim` follow curried function types.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): type checker with stable reason codes`.

---

### Task 4: Interpreter

**Files:** Create `tensegrity/symbolic/interpreter.py`; Test `tests/test_symbolic_language.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: `class Result(NamedTuple): value; steps: int`; `evaluate(e: Expr, *, fuel: int = 100_000, int_bound: int = 10**6) -> Result` (expression must be closed and well-typed; function results are returned as an opaque closure); `apply_program(program: Expr, args: Sequence[Expr], *, fuel=..., int_bound=...) -> Result` (evaluates `program`, applies it to evaluated `args`; an `Agent` takes one arg per input); `render_value(v, t: Type) -> str` (ints `3`, bools `true`/`false`, lists `(list 1 2)`, empty list `(nil T)`); `EvalError` codes: `fuel`, `int_bound`, `empty_list`.

- [ ] **Step 1: Write failing tests**: `test_spec_compose_example` (`apply_program(parse("(compose (add 2) (mul 3))"), [parse("4")]).value == 14`... note: `(compose f g) x = f(g(x))` so `add2(mul3(4)) = 14`; assert 14); `test_steps_are_deterministic_and_counted` (same expression twice -> equal `steps`, `steps > 0`); `test_fold_map_filter` (sum 6, map [2,3,4], filter `(lt 2)` keeps [3]... assert exact values); `test_agent_application`; `test_head_of_empty_is_evalerror_code_empty_list`; **Review Focus 5**: `test_int_bound_overflow` (`(mul 1000000 1000000)` -> `EvalError` code `int_bound`), `test_fuel_exhaustion` (`fuel=10` on a 100-element fold -> code `fuel`); `test_partial_application_values`; `test_no_python_exception_leaks` (2000 random typed expressions from the Task 2 test generator evaluated under tiny fuel: only `EvalError` may be raised).
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** a tree-walking evaluator with explicit environment dicts, a step counter incremented per node evaluation and per primitive application, and int-bound check on every arithmetic result. Lists are Python tuples; closures are a small class.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): pure interpreter with fuel and int bound`.

---

### Task 5: PRNG and SymbolicConfig

**Files:** Create `tensegrity/symbolic/prng.py`, `config.py`; Test `tests/test_symbolic_generation.py`

**Interfaces:**
- Produces (`prng.py`): `class SplitMix64: __init__(seed: int); next_u64() -> int; random() -> float` (53-bit, in [0,1)); `randrange(n: int) -> int` (rejection-sampled, unbiased); `choice(seq)`; `weighted_choice(items, weights)`. `document_rng(seed: int, split: str, index: int) -> SplitMix64` using `int.from_bytes(sha256(f"tensegrity:v1:{seed}:{split}:{index}".encode()).digest()[:8], "big")`.
- Produces (`config.py`): `@dataclass(frozen=True) class SymbolicConfig` with fields `seed=0, max_depth=4, enabled_ops` (tuple of op names; default all of §2.2), `stage_weights={"A":2,"B":1,"C":2,"D":2,"E":1}` (stored as sorted tuple of pairs for hashability, accepts dict), `n_train, n_val, n_test_comp`, `held_out_pairs`, `error_policy="keep_labelled"`, `int_bound=10**6`, `fuel=100_000`, `shard_size=256`, `synth_examples=4`, `synth_holdout=4`, `pair_focus_prob=0.15`, `max_candidates=None`. Methods `validate() -> None` (raises `ConfigError`), `to_dict()`, `from_dict(d)`, `fingerprint() -> str` (sha256 of sorted-key JSON).

- [ ] **Step 1: Write failing tests**: `test_splitmix_known_vector` (seed 0 -> first three `next_u64()` equal the published SplitMix64 outputs `0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F`); `test_document_rng_is_independent_of_other_indices` (value for index 50 equal whether or not indices 0-49 were drawn); `test_randrange_uniform_enough` (n=7, 70 000 draws, each bucket within 5% of 10 000); `test_config_roundtrip_and_fingerprint_stable`; `test_config_validate_rejects` (held-out op not in `enabled_ops`, negative counts, unknown stage letter, unknown `error_policy`).
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** both modules.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): splitmix64 PRNG and SymbolicConfig`.

---

### Task 6: Type-directed sampler with productivity table

**Files:** Create `tensegrity/symbolic/sampler.py`; Test `tests/test_symbolic_generation.py`

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `SAMPLE_TYPES: Tuple[Type, ...]` (the universe in "Decisions"); `productive(cfg: SymbolicConfig) -> Dict[Tuple[Type, int], bool]` (can type `t` be built at depth <= d from `cfg.enabled_ops`, literals and variables in scope); `sample_expr(rng: SplitMix64, t: Type, depth: int, cfg: SymbolicConfig, env: Optional[Dict[str, Type]] = None, force_pair: Optional[Tuple[str, str]] = None) -> Expr` (raises `ConfigError` if `(t, depth)` is not productive; with `force_pair` the root node is the pair's parent with the child operator as the relevant direct child); `sample_value(rng, t: Type) -> Expr` (literal of ground type: ints in [-9, 9], lists of length 0-5).

- [ ] **Step 1: Write failing tests**: `test_every_sample_typechecks_to_requested_type` (all `SAMPLE_TYPES` x depths 1-4 x 300 seeds with default config: `infer(e) == t`, `depth(e) <= depth`); `test_infeasible_config_raises` (`enabled_ops=("add",)` requesting `Bool` -> `ConfigError`... note `true`/`false` literals are always allowed, so request `(-> Int Bool)` instead, which is then infeasible); `test_force_pair_produces_pair` (for each default held-out pair, `collect_pairs` is defined in Task 7, so here assert via `children`/node class: `("map","compose")` root is `MapE` with a `Compose` first child); `test_sampling_is_deterministic` (same rng seed -> same serialized program); `test_sample_value_types`.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** the productivity table as a fixpoint over depth levels; the sampler picks uniformly among productive constructors for `(t, depth)` using the supplied PRNG only. Depth 1 means a literal, variable or bare primitive.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): type-directed sampler with feasibility analysis`.

---

### Task 7: Operator pairs and split assignment

**Files:** Create `tensegrity/symbolic/splits.py`; Test `tests/test_symbolic_generation.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 5.
- Produces: `op_name(e: Expr) -> Optional[str]` (for `App` with `Prim`/`Var` head: that name; `Prim(name)`: name; special forms: `lambda let if compose map filter fold list agent`; `Lit/Var/Nil/Hole`: `None`); `collect_pairs(e: Expr) -> FrozenSet[Tuple[str, str]]` (ordered parent->direct-child where both have an `op_name`); `assign_split(programs: Sequence[Expr], cfg: SymbolicConfig) -> str` returning `"test_comp"` if any program's pairs intersect `cfg.held_out_pairs`, else `"train"` or `"val"` by `int.from_bytes(sha256(f"{cfg.seed}:{serialize(programs[0])}".encode()).digest()[:8],"big") % 10000 < 9000`... use `train_threshold = round(10000 * n_train / (n_train + n_val))`.

- [ ] **Step 1: Write failing tests**: `test_pairs_for_spec_example` (`(map (compose (add 1) (mul 2)) (list 1))` contains `("map","compose")` and `("compose","add")`, `("compose","mul")`); `test_partial_application_child_uses_prim_name` (`(fold add 0 xs)` child pair `("fold","add")`); `test_heldout_goes_to_test_comp_regardless_of_hash` (1000 programs containing a held-out pair -> all `test_comp`); `test_nonheldout_split_is_stable_and_seed_dependent`; `test_ratio_matches_config` (10 000 programs without held-out pairs -> train fraction within 2% of `n_train/(n_train+n_val)`); `test_any_program_in_document_triggers_test_comp` (second program in the list holds the pair).
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** both functions; hash only on `programs[0]`.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): operator-pair detection and split assignment`.

---

### Task 8: Stage builders and mutation pass

**Files:** Create `tensegrity/symbolic/stages.py`; Test `tests/test_symbolic_generation.py`

**Interfaces:**
- Consumes: Tasks 1-7.
- Produces: `@dataclass class Doc: text: str; stage: str; programs: Tuple[str, ...]; result_type: str; expected: Optional[str]; structure_required: bool; difficulty: Dict[str, int]; extra: Dict[str, Any]` with `to_record() -> Dict[str, Any]`; `build_doc(stage: str, rng: SplitMix64, cfg: SymbolicConfig, force_pair: Optional[Tuple[str,str]] = None) -> Optional[Doc]` (returns `None` when the candidate is rejected, e.g. `EvalError` under `error_policy="reject"`); `MUTATIONS: Tuple[str, ...] = ("drop_paren", "unknown_op", "arity", "type_mismatch", "unbound_var")` and `mutate(program_text: str, rng, kind: str) -> str`.
- Text formats exactly as spec §4.3 (`CHECK <p>\nOK <type>` etc.: header line, newline, answer line). Stage A mutants are only produced from programs with no held-out pair; their label is the real `check_text` result. Stage C programs are functions or agents; input is a literal from `sample_value`; `EvalError` under `keep_labelled` yields answer `ERR eval: <code>`. Stage D stores `extra["holdout"] = [[in_text, out_text], ...]` (`cfg.synth_holdout` pairs) and `extra["target"]`. Difficulty keys: `depth`, `nodes`, `pairs` (count of distinct pairs), `steps` (interpreter steps, 0 for stage A).

- [ ] **Step 1: Write failing tests**: `test_stage_texts_match_spec_shapes` (one per stage: starts with `CHECK `/`COMPLETE `/`EVAL `/`SYNTH `/`COMPOSE `); `test_stage_c_expected_matches_interpreter` (re-parse `text`, rerun, equal `expected`) over 500 seeds; `test_stage_a_label_is_checker_result` (all mutation kinds, 300 seeds each: answer line equals `check_text(mutant)`); `test_mutation_never_creates_heldout_pair` (every parseable mutant's `collect_pairs` disjoint from held-out when the source was); `test_stage_d_reference_satisfies_examples_and_holdout` (running `extra["target"]` on each shown and held-out input reproduces the stated output); `test_stage_e_programs_include_operands_and_result`; `test_error_policy` (force an overflowing program via a fixed rng search: `keep_labelled` -> doc with `ERR eval: int_bound`, `reject` -> `None`); `test_text_has_no_eos_or_non_ascii`.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** all stages. Stage B: replace one random subexpression with `_` and use `serialize` for both texts. Stage E: sample two `(-> Int Int)` operands and build `Compose`; the answer shows the canonical composed program then a second line `EVAL <composed> @ <int>` result. Stage D uses `cfg.synth_examples` shown pairs; `structure_required` is true iff `force_pair` was used.
- [ ] **Step 4: Run** -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): curriculum stage builders and mutation pass`.

---

### Task 9: Corpus generation: shards, assembly, resume

**Files:** Create `tensegrity/symbolic/corpus.py`; Test `tests/test_symbolic_corpus.py`

**Interfaces:**
- Consumes: Tasks 5-8.
- Produces: `generate_shard(cfg: SymbolicConfig, shard_index: int) -> List[Dict[str, Any]]` — pure; candidate indices `shard_index*shard_size ...`; each candidate: `rng = document_rng(cfg.seed, "pool", i)`, stage drawn by `weighted_choice`, with probability `pair_focus_prob` a `force_pair` cycles through `held_out_pairs`; accepted docs become records with `id`-less fields plus `candidate_index`, `split` (from `assign_split`), `canonical_hash` (sha256 of `"\n".join(programs)`). `generate_corpus(cfg: SymbolicConfig, out_dir: str, *, resume: bool = True) -> Dict[str, Any]` (returns the manifest dict). Files written: `train.jsonl`, `val.jsonl`, `test_comp.jsonl` (one JSON object per line, `sort_keys=True`, `ensure_ascii=True`; fields per spec §5; `id = f"{split}-{accepted_index:06d}"`) and `manifest.json` (spec §5 contents + `corpus_hash`, per-file sha256). Work dir `out_dir/_work/` holds `shard-{n:06d}.jsonl` (first line a header `{"fingerprint": cfg.fingerprint(), "shard": n}`), written to `*.tmp` then `os.replace`d; removed after a successful assembly.
- Assembly: iterate shards in order; drop records whose full `text` was already seen in the same split, or any of whose programs is already owned by a different split; fill each split to its quota; stop when all quotas are met; raise `CorpusError` (message names split, have/want, candidates used) if `max_candidates` is reached first.

- [ ] **Step 1: Write failing tests**: `test_same_seed_byte_identical` (two output dirs, compare all four files' bytes); `test_resume_matches_uninterrupted` (generate shards 0-2 only via `generate_shard` + manual write to `_work`, then `generate_corpus(resume=True)` equals a clean run); `test_shard_is_pure` (`generate_shard` twice equal, and independent of other shards); `test_quota_counts_exact` (record counts equal `n_train/n_val/n_test_comp`); `test_no_canonical_overlap_across_splits`; `test_no_heldout_pair_in_train_or_val` (re-parse via `parse` and `collect_pairs`); **Review Focus 2**: `test_resume_refuses_different_config` (second run with different seed into same dir with leftover `_work` -> `CorpusError` mentioning `fingerprint`); **Review Focus 3**: `test_corrupt_and_tmp_shards_are_regenerated` (leave a `.tmp` file and a shard whose last line is truncated; result still equals a clean run); **Review Focus 4**: `test_unfillable_quota_fails_loudly` (`n_test_comp=10_000`, `max_candidates=500` -> `CorpusError` containing `test_comp`); `test_manifest_hashes_match_files`; `test_stage_mix_matches_weights` (train split, `n_train=4000`: each stage fraction within 0.03 of normalised weight).
- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/test_symbolic_corpus.py -q` -> FAIL.
- [ ] **Step 3: Implement** as specified; corrupt-shard detection = header mismatch or any line failing `json.loads` -> regenerate that shard; fingerprint mismatch on any existing shard header -> `CorpusError`.
- [ ] **Step 4: Run** -> PASS. Also run once with `n_train=5000,n_val=500,n_test_comp=500` and print wall time (benchmark, not gated).
- [ ] **Step 5: Commit** `feat(symbolic): deterministic resumable corpus generation`.

---

### Task 10: Corpus auditor

**Files:** Create `tensegrity/symbolic/verification.py`; Test `tests/test_symbolic_corpus.py`

**Interfaces:**
- Consumes: Tasks 2-9.
- Produces: `@dataclass class AuditReport: ok: bool; problems: List[str]; stats: Dict[str, Any]`; `audit_corpus(out_dir: str) -> AuditReport`. Independent of generator internals: reads `manifest.json` and the three JSONL files; extracts every top-level balanced S-expression span from each `text` with its own scanner (unparseable spans are skipped; stage A mutants cannot introduce pairs, see Task 8), parses them, and checks: no held-out pair in train/val; zero canonical-program overlap across splits; recomputes every `OK`/`ERR` label (via `check_text`) and every `expected` value (via the interpreter) for stages A and C; `text` contains no `<|eos|>`, no byte 257 equivalent, no NUL; per-split stage quotas within 0.03 (train only) of manifest weights; no duplicate `text` within a split; file hashes equal manifest.

- [ ] **Step 1: Write failing tests**: `test_clean_corpus_passes`; one test per tamper, each asserting `report.ok is False` and the matching message substring: inject a held-out pair doc into `train.jsonl`; copy a train program into `val.jsonl`; flip an `expected`; flip an `OK` label; append a duplicate line; edit a file without updating the manifest hash.
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** `audit_corpus`; `generate_corpus` calls it before returning and raises `CorpusError` if `not report.ok`.
- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/test_symbolic_corpus.py -q` -> PASS.
- [ ] **Step 5: Commit** `feat(symbolic): independent corpus auditor`.

---

### Task 11: CLI, ingestion compatibility, version 2.3.0, docs

**Files:**
- Modify: `tensegrity/cli.py` (new `cmd_symbolic_corpus` near `cmd_prepare` at line ~120, parser registered next to `prepare` at line ~387), `tensegrity/__init__.py:10` and `pyproject.toml:7` (`2.3.0`), `tensegrity/symbolic/__init__.py` (re-exports), `README.md` (short "Symbolic corpus" section)
- Test: `tests/test_symbolic_corpus.py`, existing `tests/test_cli_extras.py` / `tests/test_installation_cli.py` must still pass

**Interfaces:**
- Consumes: Tasks 9-10.
- Produces: `tensegrity symbolic-corpus --out DIR [--seed N] [--config FILE.json] [--n_train N --n_val N --n_test_comp N] [--no_resume] [--audit_only]`; `--audit_only` runs `audit_corpus(DIR)` and exits 1 when `not ok`. Defaults: `n_train=20000, n_val=1000, n_test_comp=1000`. Prints counts, corpus hash and audit result.

- [ ] **Step 1: Write failing tests**: `test_cli_generates_and_audits` (subprocess `[sys.executable, "-m", "tensegrity", "symbolic-corpus", "--out", d, "--n_train", "200", "--n_val", "20", "--n_test_comp", "20"]`, exit 0; then `--audit_only` exit 0; after corrupting a line, `--audit_only` exit 1); **ingestion criterion**: `test_prepare_data_yields_one_eos_per_document` (generate small corpus, `prepare_data(train.jsonl, out, text_field="text", val_fraction=0.1, min_documents=1)`; count of `ByteTokenizer.EOS` in `train.bin`+`val.bin` equals number of records; pass `split="tail"` if the document split refuses); `test_version_is_2_3_0` (`tensegrity.__version__ == "2.3.0"` and pyproject text contains `version = "2.3.0"`); `test_symbolic_package_imports_without_torch` (subprocess with `sys.modules['torch']=None` poisoning then `import tensegrity.symbolic` succeeds — if `tensegrity/__init__.py` is lazy this holds; otherwise import via file path and say so in the test).
- [ ] **Step 2: Run** -> FAIL.
- [ ] **Step 3: Implement** CLI wiring following the `cmd_prepare` / `sub.add_parser(...)` pattern (lazy imports inside `cmd_*`); bump versions; add README section (generation command, file layout, EOS note).
- [ ] **Step 4: Run the whole suite** `.venv/bin/python -m pytest -q`. Expected: all pass except the known `tests/test_notebook.py` failures (identical to the pre-change baseline recorded by running the suite once at the start of Task 1; if any other test fails, fix before committing). Note: `tests/test_packaging.py` builds the deb from `rglob("*.py")`, so the new package is picked up automatically.
- [ ] **Step 5: Commit** `feat(symbolic): symbolic-corpus CLI, ingestion test, 2.3.0`.

---

## Self-review notes

- **Spec coverage:** §2 language -> Tasks 1-4; §3 layout -> File Structure (with the two documented splits); §4.1 -> Tasks 5, 9; §4.2 -> Task 6; §4.3 -> Task 8; §4.4 -> Tasks 7, 9; §4.5 -> Task 8 (`extra`, `structure_required`; scoring itself is sub-project 2); §5 -> Tasks 9, 11; §6 -> Task 10; §7 criteria -> tests named in Tasks 2-11 (throughput benchmark is Task 9 step 4; quota test in Task 9; ingestion in Task 11).
- **Types:** `SplitMix64`, `document_rng`, `SymbolicConfig`, `Doc`, `collect_pairs`, `assign_split`, `check_text` are defined once and used with the same names later.
- **Spec deviations to confirm with the author:** `"pool"` replaces `{split}` in the RNG seed string; `splits.py`/`stages.py` extracted from `corpus.py`; extra config fields; Task 1 records the test baseline.
