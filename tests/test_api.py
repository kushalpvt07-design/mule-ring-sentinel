"""
tests/test_api.py
─────────────────
The serving path. Two halves, because they can run in different places.

─────────────────────────────────────────────────────────────────────────────
WHY THIS FILE IS SPLIT
─────────────────────────────────────────────────────────────────────────────
`api/main.py` imports FastAPI, xgboost, pandas and networkx, and scoring anything
needs a trained booster and a context graph on disk. On a bare checkout — a
reviewer's laptop, a CI container that installs nothing — none of that is present,
and a file that skips wholesale there defends nothing at the moment it is most
likely to be read.

So the structural guarantees are asserted with `ast` against the source text,
importing nothing at all. They cannot check that a request returns 200; they can
check the things that no amount of green behavioural testing would notice, because
they are properties of the code's shape:

  • every successful exit from /score passes through `validate_response_batch`
  • the endpoint's error states refuse rather than degrade
  • explanations come from TreeSHAP and not from global gains
  • /health can express "I have no threshold" instead of inventing one
  • `ServiceState.ready` still requires all six of its prerequisites, and requires
    them jointly
  • nothing on the serving path defaults, or falls back to, a number it could not
    read out of metrics.json
  • the surface is exactly two routes, both declaring a response model, and a
    SHAP contribution cannot reach a client as anything looser than a
    `ContributingFactor`

Each of those claims was checked by planting the defect it names in a copy of the
source and confirming the assertion goes red — forty-one mutants over two passes,
against three controls (a renamed local, a reworded print, a reordered `and`
chain) that must leave every guard green. A guard that never fails is decoration.

That discipline has twice caught a guard of mine that was asserting nothing:

  • `_conditional_ancestors` exists because the first version of the gate-branch
    guard asked whether any statement of the endpoint body contained the call,
    which an enclosing `if` satisfies trivially. It passed a mutant that wrapped
    the gate in `if threshold is not None:` — the one defect it existed to catch.
  • `_assignments_to` unpacks parallel assignment because
    `STATE.threshold, STATE.threshold_source = threshold, source` has an
    `ast.Tuple` target. Matching only `Attribute` targets found no assignments to
    STATE.threshold at all, so the guard built on it passed by inspecting an
    empty list. It now fails if it finds nothing to inspect.

The behavioural half below needs the real stack and is `importorskip`-gated. It
runs where the model exists and asserts what the ast half cannot: actual status
codes, actual Pydantic rejections, actual tier arithmetic on a live response.

─────────────────────────────────────────────────────────────────────────────
WHAT IS DELIBERATELY NOT HERE
─────────────────────────────────────────────────────────────────────────────
`merge_with_context`'s graph reassembly is already pinned in
tests/test_contract.py (`test_merge_with_context_reassembles_the_graph_exactly`),
as is the trained-window measurement and the startup contract check. The
defense-only rails themselves — every forbidden verb, every tier boundary, batch
forgery — are tests/test_responder.py's subject. Duplicating either here would
mean two places to update and one of them going stale.

In particular the *feature contract* is tests/test_contract.py's, not this file's:
`test_api_startup_invokes_the_contract_check` walks api/main.py for the call, and
`test_no_module_hard_codes_a_feature_list` sweeps every module in the repo for a
second copy of the feature names. What section 6 below adds is the one thing
neither of those can see — that the call's *result* gates serving, rather than
being logged and then contradicted by a flag set anyway.

Usage:
    pytest tests/test_api.py -v
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAIN_PATH = PROJECT_ROOT / "api" / "main.py"
SCHEMAS_PATH = PROJECT_ROOT / "api" / "schemas.py"

# The gate. Named once so a rename is a single edit here rather than a silent
# hole in three assertions.
GATE = "validate_response_batch"
ENDPOINT = "score_transactions"


# ══════════════════════════════════════════════════════════════════
# Helpers — shape questions asked of the source, importing nothing
# ══════════════════════════════════════════════════════════════════

def _tree(path: Path) -> ast.Module:
    if not path.exists():
        pytest.fail(f"{path.relative_to(PROJECT_ROOT)} is missing")
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """The named top-level function, sync or async."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    pytest.fail(f"no top-level function named {name!r}")


def _called_names(node: ast.AST) -> list[str]:
    """Every callee name in the subtree, attribute calls as their final part."""
    out: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            out.append(func.id)
        elif isinstance(func, ast.Attribute):
            out.append(func.attr)
    return out


def _class_field_annotation(tree: ast.Module, class_name: str,
                            field: str) -> ast.expr | None:
    """The annotation expression for one field of one class, or None."""
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) \
                    and isinstance(stmt.target, ast.Name) \
                    and stmt.target.id == field:
                return stmt.annotation
        pytest.fail(f"{class_name} has no annotated field {field!r}")
    pytest.fail(f"no class named {class_name!r}")
    return None


def _admits_none(annotation: ast.expr) -> bool:
    """True if the annotation is Optional, written either way."""
    text = ast.unparse(annotation)
    return "None" in text or text.startswith("Optional[")


def _field_call_source(tree: ast.Module, class_name: str, field: str) -> str:
    """
    The unparsed `Field(...)` expression for one field of one class.

    Scoped to the class on purpose. Searching the whole file for `min_length`
    would be satisfied by a bound declared on some other model entirely — the
    guard would pass while the field it names went unconstrained, which is the
    substring-matching trap this repo has been bitten by before.
    """
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) \
                    and isinstance(stmt.target, ast.Name) \
                    and stmt.target.id == field:
                if stmt.value is None:
                    pytest.fail(
                        f"{class_name}.{field} has no assigned value, so it "
                        f"carries no Field() constraints at all")
                return ast.unparse(stmt.value)
        pytest.fail(f"{class_name} has no annotated field {field!r}")
    pytest.fail(f"no class named {class_name!r}")
    return ""


def _http_statuses(node: ast.AST) -> list[int]:
    """Every literal `status_code=` on an HTTPException raised in the subtree."""
    out: list[int] = []
    for child in ast.walk(node):
        if not (isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call)):
            continue
        callee = child.exc.func
        name = callee.id if isinstance(callee, ast.Name) else \
            getattr(callee, "attr", "")
        if name != "HTTPException":
            continue
        for kw in child.exc.keywords:
            if kw.arg == "status_code" and isinstance(kw.value, ast.Constant):
                out.append(int(kw.value.value))
    return out


# Constructs that make the code inside them reachable on some paths and not
# others. `BoolOp` and `IfExp` are here because `x = cached or gate(...)` and
# `x = gate(...) if strict else x` are conditional bypasses written as one
# statement, and a check that only looked at statement types would wave both
# through. `Try` is here because a narrow `except` around the gate is a bypass
# even though `test_the_endpoint_swallows_nothing` only forbids the broad ones.
_CONDITIONAL = (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Try,
                ast.With, ast.AsyncWith, ast.ExceptHandler, ast.BoolOp,
                ast.comprehension, ast.Lambda)


def _conditional_ancestors(scope: ast.AST, target: ast.AST) -> list[str]:
    """
    Names of the branching constructs between `target` and `scope`, outermost last.

    Empty means `target` runs on every path through `scope`. Built from a parent
    map rather than by checking which statement of `scope.body` contains the
    target: an `ast.If` sitting in `scope.body` *does* contain it, so a
    containment test reports a conditional call as top-level. That was the first
    version of this helper, and a mutant that wrapped the gate in
    `if threshold is not None:` passed it.
    """
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(scope):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    found: list[str] = []
    current = target
    while current is not scope:
        current = parents[current]
        if isinstance(current, _CONDITIONAL):
            found.append(type(current).__name__)
    return found


def _method(tree: ast.Module, class_name: str, name: str) -> ast.FunctionDef:
    """One method (or property) of one class."""
    for node in tree.body:
        if not (isinstance(node, ast.ClassDef) and node.name == class_name):
            continue
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and stmt.name == name:
                return stmt
        pytest.fail(f"{class_name} has no method named {name!r}")
    pytest.fail(f"no class named {class_name!r} in the module")
    raise AssertionError                                   # unreachable


def _self_attributes(scope: ast.AST) -> set[str]:
    """Every `self.<attr>` read or written in the subtree, as bare attr names."""
    return {node.attr for node in ast.walk(scope)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == "self"}


def _assignments_to(scope: ast.AST, owner: str,
                    attr: str) -> list[tuple[int, ast.expr]]:
    """
    Every value assigned to `owner.attr` in the subtree, as (line, expression).

    Parallel assignments are unpacked. `STATE.threshold, STATE.threshold_source =
    threshold, source` at api/main.py:452 has an `ast.Tuple` for its target, so a
    version of this that only looked at `Attribute` targets reported no
    assignments to STATE.threshold at all — and every guard built on it passed by
    inspecting nothing. Unpacking is what makes those guards non-vacuous.
    """
    out: list[tuple[int, ast.expr]] = []

    def pairs(target: ast.expr, value: ast.expr):
        if isinstance(target, (ast.Tuple, ast.List)) \
                and isinstance(value, (ast.Tuple, ast.List)) \
                and len(target.elts) == len(value.elts):
            for sub_target, sub_value in zip(target.elts, value.elts):
                yield from pairs(sub_target, sub_value)
        elif isinstance(target, ast.Attribute) and target.attr == attr \
                and isinstance(target.value, ast.Name) \
                and target.value.id == owner:
            yield value

    for node in ast.walk(scope):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            for value in pairs(target, node.value):
                out.append((node.lineno, value))
    return out


def _get_calls_with_defaults(scope: ast.AST) -> list[tuple[str, str]]:
    """
    Every `<something>.get(key, default)` in the subtree, as (key, default).

    Two positional arguments only — a one-argument `.get` returns None on a
    missing key, which is the behaviour these tests want to see preserved.
    """
    out: list[tuple[str, str]] = []
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and len(node.args) == 2):
            continue
        key = node.args[0]
        out.append((key.value if isinstance(key, ast.Constant) else
                    ast.unparse(key), ast.unparse(node.args[1])))
    return out


def _route_decorators(tree: ast.Module) -> list[tuple[str, str, ast.Call]]:
    """
    Every `@app.<verb>("/path", ...)` in the module, as (verb, path, call).

    Reads the decorator list rather than the live `app.routes`, so it needs no
    FastAPI. The cost is that it sees only routes declared this way: a
    `app.add_api_route(...)` call or an included router would be invisible, which
    is why the inventory test below also forbids those by name.
    """
    verbs = {"get", "post", "put", "patch", "delete", "head", "options", "trace"}
    out: list[tuple[str, str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not (isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in verbs):
                continue
            path = decorator.args[0].value if (
                decorator.args and isinstance(decorator.args[0], ast.Constant)
            ) else "<computed>"
            out.append((decorator.func.attr.upper(), path, decorator))
    return out


# ══════════════════════════════════════════════════════════════════
# 1. No successful response leaves /score without passing the gate
# ══════════════════════════════════════════════════════════════════

class TestEverySuccessfulExitPassesTheGate:
    """
    The one property the defense-only design rests on at serving time.

    tests/test_responder.py proves `validate_response_batch` rejects a forged
    response. That is worth nothing if the endpoint can return without calling it.
    Every other test in the repo would stay green: the scores would be right, the
    tiers would be right, and the single unvalidated path — a new early return
    added for a fast case, say — would be the one nobody looked at.
    """

    def test_the_endpoint_calls_the_gate(self):
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)
        assert GATE in _called_names(endpoint), (
            f"{ENDPOINT} does not call {GATE}. Responses would reach callers with "
            f"whatever action assembled them, and the batch validator would be "
            f"dead code that tests/test_responder.py exercises in isolation."
        )

    def test_the_gate_result_is_what_gets_returned(self):
        """
        Calling it is not enough — the return value has to be used.

        `validate_response_batch(node_scores, ...)` on its own line, with the
        result discarded, still raises on a forgery and so still passes the test
        above. But it also permits the gate to be reduced to an assertion over a
        list that is then rebuilt, or reordered, before being served. Requiring the
        call to be an assignment to the name that is returned closes that.
        """
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)

        assigned: set[str] = set()
        for node in ast.walk(endpoint):
            if not isinstance(node, ast.Assign):
                continue
            if not (isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == GATE):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)

        assert assigned, (
            f"{GATE}'s return value is discarded in {ENDPOINT}. It must be bound "
            f"to the name that is served, so the validated list is the one that "
            f"leaves the process."
        )

        returns = [n for n in ast.walk(endpoint) if isinstance(n, ast.Return)
                   and n.value is not None]
        assert returns, f"{ENDPOINT} has no value-returning statement"

        for ret in returns:
            served = {n.id for n in ast.walk(ret) if isinstance(n, ast.Name)}
            assert served & assigned, (
                f"the return at line {ret.lineno} does not mention any name bound "
                f"from {GATE} (bound: {sorted(assigned)}). A response assembled "
                f"from anything else has not been through the gate."
            )

    def test_the_gate_is_not_inside_a_branch(self):
        """
        Unconditional: the call must run on every path through the endpoint.

        A gate under `if not request.threshold_override:` — or inside the loop that
        builds the responses, or in an `else`, or short-circuited behind an `or` —
        is a gate with a documented bypass.

        This is the guard mutation testing caught. The first version asked whether
        any statement of `endpoint.body` contained the call, which an enclosing
        `if` satisfies trivially, since the `if` is itself a statement of
        `endpoint.body`. It passed a mutant that wrapped the gate in
        `if threshold is not None:` — the exact defect it was written to catch.
        """
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)
        calls = [n for n in ast.walk(endpoint)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == GATE]
        assert calls, f"{ENDPOINT} does not call {GATE} at all"

        for call in calls:
            guards = _conditional_ancestors(endpoint, call)
            assert not guards, (
                f"the {GATE} call at line {call.lineno} sits inside {guards}. "
                f"Whatever condition that introduces is a path on which a "
                f"response is served unvalidated."
            )

    def test_there_is_exactly_one_success_exit(self):
        """
        One return, so the guard above cannot be satisfied while a second path
        quietly serves something else.

        Refusals are `raise HTTPException` and are unaffected — this counts only
        exits that hand a body to the caller.
        """
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)
        returns = [n for n in ast.walk(endpoint)
                   if isinstance(n, ast.Return) and n.value is not None]
        assert len(returns) == 1, (
            f"{ENDPOINT} has {len(returns)} value-returning statements (lines "
            f"{[n.lineno for n in returns]}). Each is a separate path that must be "
            f"shown to pass the gate; keep it to one and the proof stays local."
        )


# ══════════════════════════════════════════════════════════════════
# 2. The endpoint refuses rather than degrades
# ══════════════════════════════════════════════════════════════════

class TestUnscorableRequestsAreRefused:
    """
    Three states in which an honest score is impossible, and all three must be
    errors.

    The tempting alternative is a 200 carrying a warning string. That was the
    previous behaviour for the empty-context case and it is worse than useless:
    `risk_level` and `action` are required fields on NodeRiskScore, so the caller
    receives a complete-looking, authoritative-looking response computed on a
    four-edge graph, and the warning is in a list nobody parses.
    """

    def test_all_three_refusal_codes_are_present(self):
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)
        statuses = set(_http_statuses(endpoint))
        for code, why in (
            (503, "a prerequisite is missing and the service cannot score"),
            (422, "the context trim left no historical edges, so the graph is "
                  "not the one the threshold was calibrated on"),
            (500, "the extractor has diverged from the feature contract"),
        ):
            assert code in statuses, (
                f"{ENDPOINT} never raises HTTPException({code}), which is how it "
                f"reports that {why}. Found {sorted(statuses)}."
            )

    def test_the_empty_context_check_precedes_feature_computation(self):
        """
        Order matters, not just presence.

        Computing features first and refusing afterwards is the same refusal but
        pays for PageRank and cycle enumeration over the merged graph to reach it,
        and — worse — puts a scoring path between the two, which is where someone
        later adds "well, we already have the scores, so return them".
        """
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)

        refusal_line = None
        for node in ast.walk(endpoint):
            if isinstance(node, ast.Raise) and 422 in _http_statuses(node):
                refusal_line = node.lineno
                break
        assert refusal_line is not None, "no 422 refusal found"

        for callee in ("compute_node_features", "predict_proba"):
            lines = [n.lineno for n in ast.walk(endpoint)
                     if isinstance(n, ast.Call) and callee in _called_names(n)]
            assert lines, f"{ENDPOINT} never calls {callee}"
            assert refusal_line < min(lines), (
                f"the 422 refusal is at line {refusal_line}, after {callee} at "
                f"line {min(lines)}. The request is refused only once the work has "
                f"been done, and a scored-but-unreturnable batch is sitting in "
                f"scope inviting an early return."
            )

    def test_the_endpoint_swallows_nothing(self):
        """
        No `except Exception` in the endpoint itself.

        Scoped deliberately to `score_transactions`. `attribute` has one, and it is
        correct there: it degrades explanations to a warning while the scores go
        out intact, which is the right trade when a fraud alert with no reason is
        recoverable and one with a fabricated reason is not. The same construct in
        the endpoint would convert any of the three refusals above into whatever
        the handler decided to return.
        """
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)
        broad = []
        for node in ast.walk(endpoint):
            if not isinstance(node, ast.ExceptHandler):
                continue
            caught = ast.unparse(node.type) if node.type else "bare except"
            if caught in ("Exception", "BaseException", "bare except"):
                broad.append((node.lineno, caught))
        assert not broad, (
            f"{ENDPOINT} catches {broad}. A blanket handler here turns a refusal "
            f"into a response and there is no status code left that says the score "
            f"is not trustworthy."
        )


# ══════════════════════════════════════════════════════════════════
# 3. Explanations are attributions, not global gains
# ══════════════════════════════════════════════════════════════════

class TestExplanationsComeFromShap:
    """
    The v2 defect this repo documents at the top of api/main.py: the endpoint
    filled `top_features` with the model's three highest global gains, so every
    alert in every batch carried the same three names. It described the model, not
    the account, and it looked entirely plausible in the response body.
    """

    def test_the_endpoint_attributes_per_account(self):
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)
        assert "attribute" in _called_names(endpoint), (
            "the endpoint does not call `attribute`, so whatever fills "
            "contributing_factors is not a per-account SHAP attribution."
        )

    def test_no_serving_path_reads_global_gains(self):
        """
        `feature_importances_` is a property of the booster, identical for every
        row. Reading it anywhere on the scoring path means an explanation that
        cannot vary by account.
        """
        tree = _tree(MAIN_PATH)
        offenders = [
            node.lineno for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr in ("feature_importances_", "get_score")
        ]
        assert not offenders, (
            f"api/main.py reads global feature importance at line(s) {offenders}. "
            f"That is the same three names on every alert regardless of the "
            f"account, which is what v2 shipped."
        )

    def test_the_attribution_helper_verifies_the_summation_identity(self):
        """
        Exact TreeSHAP guarantees `sum(contributions) + bias` equals the model's
        raw MARGIN. The check is made there and not one sigmoid later, because a
        fixed probability tolerance is a varying margin tolerance — near p = 0.9999
        the old form tolerated over 1.5 log-odds of attribution error, and past its
        ±60 clip it tolerated anything. `models/train.py` was already correct on
        this; the serving copy was the loose one.

        The tolerance is required BY NAME, and the name matters: MARGIN, not
        probability. Renaming it back to a probability-space constant is the exact
        regression this asserts against.
        """
        helper = _function(_tree(MAIN_PATH), "attribute")
        source = ast.unparse(helper)
        assert "MARGIN_IDENTITY_TOLERANCE" in source, (
            "`attribute` no longer compares the reconstructed MARGIN against "
            "MARGIN_IDENTITY_TOLERANCE. Attributions that do not sum to the "
            "model's own margin do not explain it, and shipping them sends an "
            "analyst somewhere the model never went."
        )
        assert "output_margin" in source, (
            "`attribute` reconstructs the identity without asking the model for "
            "its margin. `predict(output_margin=True)` applies the same "
            "early-stopping iteration range predict_proba does; a hand-rolled "
            "log(p / (1 - p)) does not, and is worst precisely at the saturated "
            "scores where CRITICAL accounts live."
        )
        assert "ContributingFactor" in _called_names(helper), (
            "`attribute` no longer builds ContributingFactor objects; raw "
            "contribution arrays are not analyst-readable."
        )

    def test_the_attribution_helper_checks_the_bias_column_separately(self):
        """
        THE CLAIM THIS FILE USED TO MAKE AND COULD NOT SUPPORT.

        The old version of the test above asserted the summation identity and said
        in its docstring that it caught "mistaking the trailing bias column for a
        feature". It does not, and cannot. `pred_contribs` returns n_features + 1
        columns whose row sum is the margin, so `contribs.sum(1) + bias` recovers
        that sum for ANY split of the columns — `raw[:, :-1] / raw[:, -1]` and
        `raw[:, 1:] / raw[:, 0]` give bit-identical totals. Under the off-by-one
        every score stays correct and every factor NAME shifts one position, which
        is the worst failure available to an analyst-facing explanation and the one
        the guard advertised catching.

        What separates the bias column from a feature is that it is constant down
        the batch — a scalar `base_score` repeated. So the spread of `bias` must be
        zero, and is checked as its own gate.
        """
        helper = _function(_tree(MAIN_PATH), "attribute")
        source = ast.unparse(helper)
        assert "ptp" in source, (
            "`attribute` does not check that the TreeSHAP bias column is constant "
            "across the batch. Without it nothing in the serving path can detect "
            "the bias column being read from the wrong end of pred_contribs — the "
            "summation identity is invariant to that split, so it passes either "
            "way while every feature name shifts by one."
        )
        assert "BIAS_CONSTANT_TOLERANCE" in source, (
            "the bias-spread check no longer compares against "
            "BIAS_CONSTANT_TOLERANCE. A bare `== 0` would fail on float32 "
            "accumulation noise; a missing bound is not a check."
        )


# ══════════════════════════════════════════════════════════════════
# 4. /health can say "I don't have one"
# ══════════════════════════════════════════════════════════════════

class TestHealthReportsMissingValuesAsMissing:
    """
    A health endpoint that invents a plausible number in the degraded state is
    worse than one that 500s, because the reader believes it.
    """

    @pytest.mark.parametrize("field", ["optimal_threshold",
                                       "partition_fingerprint",
                                       "model_file"])
    def test_optional_state_is_optional_in_the_schema(self, field):
        """
        `optimal_threshold` was `float = 0.5` while api/main.py passed
        `STATE.threshold`, typed `float | None` and left as None whenever the
        threshold could not be read. Under pydantic v2 that combination raises a
        ValidationError, so /health returned a 500 in precisely the degraded state
        it exists to describe — and the comment beside the call already claimed the
        field was `float | None`, so the source documented a fix it had not made.

        0.5 is the worse default of the two available: 0.0 at least names a value
        `classify_risk` rejects outright, whereas 0.5 reads like a real operating
        point.
        """
        annotation = _class_field_annotation(
            _tree(SCHEMAS_PATH), "HealthResponse", field)
        assert _admits_none(annotation), (
            f"HealthResponse.{field} is annotated "
            f"{ast.unparse(annotation)!r}, which cannot represent absence. "
            f"api/main.py populates it from ServiceState, where it is Optional."
        )

    def test_the_scoring_context_admits_an_unknown_trained_window(self):
        """
        Same shape of defect one field over: `trained_window_days` is None when the
        training edges could not be measured, and a 0.0 there would make every
        window look infinitely divergent.
        """
        annotation = _class_field_annotation(
            _tree(SCHEMAS_PATH), "ScoringContext", "trained_window_days")
        assert _admits_none(annotation), (
            f"ScoringContext.trained_window_days is "
            f"{ast.unparse(annotation)!r} and cannot say 'not measured'."
        )


# ══════════════════════════════════════════════════════════════════
# 5. The request schema constrains the batch
# ══════════════════════════════════════════════════════════════════

class TestRequestValidationIsDeclared:
    """
    Structural half of the input contract. The behavioural half below submits
    actual bad payloads; these assert the constraints exist at all, which is what
    catches a `Field(...)` quietly losing its bounds during a refactor.
    """

    def test_the_batch_is_bounded_at_both_ends(self):
        """The field must exist, and its bounds must be declared on it."""
        field = _field_call_source(_tree(SCHEMAS_PATH), "ScoringRequest",
                                   "transactions")
        for constraint, why in (
            ("min_length", "an empty batch would compute features over the "
                           "context graph alone and score nothing the caller "
                           "asked about"),
            ("max_length", "an unbounded batch is a PageRank fixpoint over an "
                           "arbitrarily large graph inside one request"),
        ):
            assert constraint in field, (
                f"ScoringRequest.transactions declares no {constraint}: {why}.\n"
                f"  Field(...) reads: {field}"
            )

    def test_a_threshold_override_cannot_be_zero(self):
        """
        `classify_risk` raises on a threshold of 0 because every non-negative score
        satisfies `>= 0`, which rates the whole population CRITICAL. Rejecting it
        at the schema turns a 500 deep in the responder into a 422 naming the
        field.
        """
        field = _field_call_source(_tree(SCHEMAS_PATH), "ScoringRequest",
                                   "threshold_override")
        assert "gt=0.0" in field or "gt=0" in field, (
            f"the threshold override is not bounded strictly above zero, so a "
            f"zero reaches classify_risk and surfaces as a 500 rather than a "
            f"validation error naming the field.\n  Field(...) reads: {field}"
        )
        assert "le=1.0" in field or "le=1" in field, (
            f"the threshold override has no upper bound; a threshold above 1 "
            f"flags nothing and so describes no operating point.\n"
            f"  Field(...) reads: {field}"
        )

    def test_self_transfers_are_rejected_at_the_edge(self):
        """
        A self-loop inflates in_degree and out_degree together and fakes a
        one-account cycle, so it is a free way to move a score. tests/
        test_leakage.py asserts the generated data contains none; this asserts the
        API will not accept one.
        """
        tree = _tree(SCHEMAS_PATH)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TransactionEdge":
                source = ast.unparse(node)
                assert "model_validator" in source, (
                    "TransactionEdge has no model_validator, so sender == "
                    "receiver is accepted."
                )
                return
        pytest.fail("no TransactionEdge class in api/schemas.py")


# ══════════════════════════════════════════════════════════════════
# 6. The service cannot call itself ready without its prerequisites
# ══════════════════════════════════════════════════════════════════

# Each prerequisite, and the state the service would be serving from if `ready`
# stopped requiring it. All six are the difference between a 503 and a 200 built
# on something the process does not have.
READY_PREREQUISITES = {
    "model": "there would be no booster to score with",
    "contract_verified": "the booster's column order would never have been "
                         "checked against FEATURE_COLS — the exact hole through "
                         "which a 12-feature v1 model was served against v3's "
                         "threshold",
    "threshold": "there would be no operating point, and `load_metrics` supplies "
                 "no default on purpose",
    "break_even_probability": "the step-up band would have no economic floor, so "
                              "the ALLOW/review line would be invented rather "
                              "than read from the cost model",
    "context_edges": "features would be computed on the submitted batch alone, "
                     "against a threshold calibrated on a 60-day graph",
    "reference_partition": "community membership would be redrawn per request, so "
                           "two identical calls could put an account in different "
                           "communities",
}


class TestReadinessIsNotAnOptimisticDefault:
    """
    `/score` opens with `if not STATE.ready: raise 503`, which makes this one
    property the whole admission control for the service. Everything else in this
    file guards what happens *after* the request is admitted; this guards whether
    it should have been.

    Asserted against the `self.<attr>` reads inside the property rather than
    against its text, because `"model" in source` is satisfied by
    `model_version` — a guard that passes while the prerequisite it names is gone.
    """

    @staticmethod
    def _ready() -> ast.FunctionDef:
        return _method(_tree(MAIN_PATH), "ServiceState", "ready")

    @pytest.mark.parametrize("prerequisite", sorted(READY_PREREQUISITES))
    def test_readiness_requires(self, prerequisite):
        consulted = _self_attributes(self._ready())
        assert prerequisite in consulted, (
            f"ServiceState.ready no longer consults {prerequisite!r}. Without it "
            f"the service will admit requests when "
            f"{READY_PREREQUISITES[prerequisite]}.\n"
            f"  ready currently reads: {ast.unparse(self._ready())}"
        )

    def test_readiness_is_a_conjunction(self):
        """
        One `or` anywhere in this expression makes a single satisfied prerequisite
        sufficient for all six. It is the cheapest possible way to turn this gate
        into decoration while leaving every name above still visible in the
        source, so the parametrized test alone would not notice.
        """
        disjunctions = [node for node in ast.walk(self._ready())
                        if isinstance(node, ast.BoolOp)
                        and isinstance(node.op, ast.Or)]
        assert not disjunctions, (
            f"ServiceState.ready contains {len(disjunctions)} `or`, so some "
            f"prerequisites are alternatives rather than requirements:\n"
            f"  {ast.unparse(self._ready())}"
        )

    def test_the_contract_flag_is_set_inside_the_checked_block(self):
        """
        `contract_verified = True` has to live in the same `try` body as the
        `assert_feature_contract` call that justifies it.

        tests/test_contract.py already asserts that api/main.py *calls* the check
        (test_api_startup_invokes_the_contract_check). That test is satisfied by a
        call whose RuntimeError is caught and whose flag is set afterwards anyway
        — the check runs, fails, is logged, and a mismatched booster is served.
        The call site and the flag have to be provably on the same path.
        """
        tree = _tree(MAIN_PATH)
        flags = _assignments_to(tree, "STATE", "contract_verified")
        true_flags = [(line, value) for line, value in flags
                      if isinstance(value, ast.Constant) and value.value is True]
        assert true_flags, (
            "nothing in api/main.py ever sets STATE.contract_verified = True, so "
            "`ready` can never be satisfied — or the flag has moved and this "
            "guard needs updating."
        )

        guarded = []
        for block in ast.walk(tree):
            if not isinstance(block, ast.Try):
                continue
            if "assert_feature_contract" not in _called_names(
                    ast.Module(body=block.body, type_ignores=[])):
                continue
            in_body = {id(node) for stmt in block.body
                       for node in ast.walk(stmt)}
            guarded += [line for line, value in true_flags
                        if id(value) in in_body]

        assert guarded, (
            "STATE.contract_verified = True is set outside the try block that "
            "calls assert_feature_contract, so a failing contract check no "
            "longer prevents the flag being set. Lines: "
            + ", ".join(str(line) for line, _ in true_flags)
        )


# ══════════════════════════════════════════════════════════════════
# 7. Nothing on the serving path invents a number it could not read
# ══════════════════════════════════════════════════════════════════

# The two functions whose whole job is to read a number or admit they could not.
# Scoped to these deliberately: `risk_counts.get("HIGH", 0)` in the endpoint is a
# legitimate two-argument .get — a tally that starts at zero — and a file-wide
# rule would forbid it.
METRICS_READERS = ("load_metrics", "read_break_even_probability")


class TestNothingIsSilentlyDefaulted:
    """
    The regression these exist for is recorded in this module's own header:
    `metrics.get("optimal_threshold", 0.5)` substituted 0.5 for a missing key,
    and with a miss costing ₹2,00,000 against ₹15,000 for a false alert the
    cost-optimal threshold is near 0.07. Every response stayed well-formed while
    most of the recall the model was tuned for was thrown away.

    Note that api/main.py's docstrings quote that bad call twice, at lines 56 and
    270, to explain why it is gone. A guard written as a substring search for
    `"optimal_threshold\", 0.5"` would therefore be tripped by the module's own
    documentation of the fix — which is why these walk the tree and only look at
    real call nodes.
    """

    @pytest.mark.parametrize("reader", METRICS_READERS)
    def test_no_metrics_key_is_read_with_a_default(self, reader):
        defaults = _get_calls_with_defaults(_function(_tree(MAIN_PATH), reader))
        assert not defaults, (
            f"{reader} reads a key with a fallback value: "
            + "; ".join(f"get({key!r}, {default})" for key, default in defaults)
            + ".\nA defaulted read cannot be distinguished downstream from a "
              "real value, so the service would score at an invented operating "
              "point instead of declining."
        )

    @pytest.mark.parametrize("reader", METRICS_READERS)
    def test_every_failure_path_returns_absence(self, reader):
        """
        Both readers signal failure by returning None — `ready` then refuses and
        /health reports the reason. A numeric literal on any of those return
        paths is the same bug as a defaulted `.get`, wearing a different shape.
        """
        offenders = []
        for node in ast.walk(_function(_tree(MAIN_PATH), reader)):
            if not isinstance(node, ast.Return) or node.value is None:
                continue
            first = (node.value.elts[0] if isinstance(node.value, ast.Tuple)
                     and node.value.elts else node.value)
            if isinstance(first, ast.Constant) and isinstance(
                    first.value, (int, float)) and not isinstance(
                    first.value, bool):
                offenders.append(f"line {node.lineno}: "
                                 f"return {ast.unparse(node.value)}")
        assert not offenders, (
            f"{reader} returns a hard-coded number rather than a value it read "
            f"or None:\n  " + "\n  ".join(offenders)
        )

    def test_the_endpoint_threshold_comes_from_state_or_the_request(self):
        """
        Two sources are legitimate — the published threshold and an explicit
        override — and nothing else is. A literal assigned to `threshold` here
        would be a second copy of an operating point that metrics.json owns, and
        the response would still declare `threshold_source: "metrics.json"`
        beside it.
        """
        endpoint = _function(_tree(MAIN_PATH), ENDPOINT)
        for node in ast.walk(endpoint):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "threshold"):
                continue
            source = ast.unparse(node.value)
            assert "STATE.threshold" in source \
                or "threshold_override" in source, (
                f"{ENDPOINT} assigns threshold = {source} at line {node.lineno}, "
                f"which is neither the published threshold nor the request's "
                f"declared override."
            )

    @pytest.mark.parametrize("attr", ["reference_partition", "context_edges",
                                      "threshold", "break_even_probability"])
    def test_a_prerequisite_that_could_not_be_loaded_is_none(self, attr):
        """
        No empty fallback. `STATE.reference_partition = {}` would satisfy
        `is not None`, so the service would report itself ready and then score
        against a partition in which every account is its own community —
        `extend_partition` would assign fresh ids to all of them, and community
        size and cross-community ratio would be constant across the batch.

        The same shape applies to the others: `context_edges = pd.DataFrame()`
        passes `is not None` and empties the trim, and a numeric fallback for
        either scalar is the defaulting bug above.
        """
        empties = (ast.Dict, ast.List, ast.Set, ast.Tuple)
        offenders = []
        inspected = _assignments_to(_tree(MAIN_PATH), "STATE", attr)
        assert inspected, (
            f"no assignment to STATE.{attr} was found in api/main.py, so this "
            f"guard is inspecting nothing. Either the attribute was renamed or "
            f"_assignments_to has stopped seeing the shape it is written in."
        )
        for line, value in inspected:
            if isinstance(value, empties) and not getattr(value, "elts", None) \
                    and not getattr(value, "keys", None):
                offenders.append(f"line {line}: empty "
                                 f"{type(value).__name__.lower()}")
            elif isinstance(value, ast.Constant) and isinstance(
                    value.value, (int, float)) and not isinstance(
                    value.value, bool):
                offenders.append(f"line {line}: literal {value.value}")
            elif isinstance(value, ast.Call) and isinstance(
                    value.func, ast.Attribute) and not value.args \
                    and value.func.attr in ("DataFrame", "Series", "dict", "list"):
                offenders.append(f"line {line}: empty {ast.unparse(value)}")
        assert not offenders, (
            f"STATE.{attr} is assigned an empty or invented value:\n  "
            + "\n  ".join(offenders)
            + f"\nAn empty container satisfies `is not None`, so `ready` would "
              f"admit requests the service cannot answer. Absence must be None."
        )


# ══════════════════════════════════════════════════════════════════
# 8. The surface is exactly two typed routes
# ══════════════════════════════════════════════════════════════════

# (verb, path) -> the response model it must declare.
EXPECTED_ROUTES = {
    ("GET", "/health"): "HealthResponse",
    ("POST", "/score"): "ScoringResponse",
}

# Ways to add a route or return past the response model without a decorator.
FORBIDDEN_ROUTE_MECHANISMS = ("add_api_route", "add_route", "include_router",
                              "mount", "websocket")
RAW_RESPONSE_TYPES = ("JSONResponse", "PlainTextResponse", "HTMLResponse",
                      "StreamingResponse", "FileResponse", "ORJSONResponse")


class TestTheSurfaceIsExactlyTwoTypedRoutes:
    """
    Everything reachable from outside, and nothing else.

    One honest caveat, because the claim is easy to overstate: FastAPI is
    constructed at api/main.py:558 without `docs_url=None`, so /docs, /redoc and
    /openapi.json are also served. That is intentional for a reviewed demo — the
    schema is the documentation — and it exposes no state beyond the shapes these
    two endpoints already declare. What these tests forbid is a *third endpoint
    of ours*: a /debug that dumps STATE, a /reload, a /partition that returns the
    frozen community map.
    """

    def test_exactly_the_two_intended_routes_are_declared(self):
        declared = {(verb, path) for verb, path, _ in
                    _route_decorators(_tree(MAIN_PATH))}
        unexpected = declared - set(EXPECTED_ROUTES)
        missing = set(EXPECTED_ROUTES) - declared
        assert not unexpected, (
            f"api/main.py exposes {sorted(unexpected)} beyond the two intended "
            f"routes. A debug or state-dumping endpoint added for local work "
            f"ships to whatever runs this file."
        )
        assert not missing, f"the intended routes {sorted(missing)} are gone"

    def test_no_route_is_added_by_any_other_mechanism(self):
        """
        The decorator scan above is only an inventory of decorators.
        `app.add_api_route(...)`, an included router or a mounted sub-app would
        all add reachable paths it cannot see, so they are refused outright
        rather than left as a blind spot.
        """
        called = _called_names(_tree(MAIN_PATH))
        found = [name for name in FORBIDDEN_ROUTE_MECHANISMS if name in called]
        assert not found, (
            f"api/main.py calls {found}, which adds routes that "
            f"test_exactly_the_two_intended_routes_are_declared cannot see. "
            f"Either declare the route with a decorator or extend that test."
        )

    @pytest.mark.parametrize("route", sorted(EXPECTED_ROUTES))
    def test_the_route_declares_its_response_model(self, route):
        """
        `response_model` is what makes the response a validated shape rather than
        whatever the function happened to return. Dropping it is how raw
        contribution arrays would reach a client: FastAPI would serialise the
        object as-is, and every schema constraint on the way out — including the
        `ContributingFactor` typing below — would stop being applied.
        """
        expected = EXPECTED_ROUTES[route]
        for verb, path, call in _route_decorators(_tree(MAIN_PATH)):
            if (verb, path) != route:
                continue
            models = [ast.unparse(kw.value) for kw in call.keywords
                      if kw.arg == "response_model"]
            assert models, (
                f"{verb} {path} declares no response_model, so the endpoint's "
                f"return value is serialised unvalidated."
            )
            assert models[0] == expected, (
                f"{verb} {path} declares response_model={models[0]}, expected "
                f"{expected}."
            )
            return
        pytest.fail(f"{route} is not declared in api/main.py")

    def test_nothing_returns_a_raw_response(self):
        """A hand-built Response body bypasses the response model entirely."""
        tree = _tree(MAIN_PATH)
        names = set(_called_names(tree)) | {
            alias.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) for alias in node.names}
        found = sorted(set(RAW_RESPONSE_TYPES) & names)
        assert not found, (
            f"api/main.py imports or constructs {found}. A raw response body is "
            f"returned verbatim, so nothing declared in api/schemas.py applies "
            f"to it."
        )

    def test_factors_are_typed_models_not_loose_containers(self):
        """
        `top_factors` returns plain dicts holding numpy-derived floats. The one
        thing standing between those and the client is this annotation: widen it
        to `list[dict]` or `Any` and pydantic stops checking that a factor names
        a real feature, carries a float contribution and declares a direction —
        and the behavioural tests that read `factor.feature` are exactly the ones
        that do not run on a bare checkout.
        """
        annotation = ast.unparse(_class_field_annotation(
            _tree(SCHEMAS_PATH), "NodeRiskScore", "contributing_factors"))
        assert annotation == "list[ContributingFactor]", (
            f"NodeRiskScore.contributing_factors is annotated {annotation!r}. "
            f"Only list[ContributingFactor] makes the contents validated."
        )

    @pytest.mark.parametrize("field,expected", [
        ("feature", "str"), ("value", "float"), ("contribution", "float"),
        ("effect", "Literal['raises_risk', 'lowers_risk']"),
    ])
    def test_each_factor_field_is_a_scalar(self, field, expected):
        """
        A `float` annotation is also the coercion that turns a numpy float32 into
        something JSON-serialisable. `Any` would pass the array through and fail
        at serialisation time, in the response, in front of the caller.

        `effect` is deliberately pinned to its `Literal` rather than to `str`,
        which is what this test first asserted and what the source is stronger
        than. Widening it to `str` costs nothing visible and loses the only check
        that a direction label is one of the two words the dashboard branches on
        — a mistyped "raises-risk" would serialise happily and read as neither.
        """
        annotation = ast.unparse(_class_field_annotation(
            _tree(SCHEMAS_PATH), "ContributingFactor", field))
        assert annotation == expected, (
            f"ContributingFactor.{field} is annotated {annotation!r}, expected "
            f"{expected!r}."
        )


# ══════════════════════════════════════════════════════════════════
# 9. Input validation, behaviourally — needs pydantic, nothing else
# ══════════════════════════════════════════════════════════════════

class TestTheSchemaRejectsBadBatches:
    """
    The declarations above, actually exercised. Needs pydantic and no model, so
    this runs on a bare checkout the moment `pip install pydantic` has happened —
    the same bar tests/test_responder.py sets.
    """

    @pytest.fixture(autouse=True)
    def _needs_pydantic(self):
        pytest.importorskip("pydantic", reason="api.schemas is a pydantic model")

    @staticmethod
    def _edge(**overrides):
        base = {"sender": "alice@upi", "receiver": "bob@upi",
                "amount": 25_000.0, "timestamp": "2025-04-01T10:00:00"}
        base.update(overrides)
        return base

    def test_a_self_transfer_is_rejected(self):
        from pydantic import ValidationError

        from api.schemas import TransactionEdge
        with pytest.raises(ValidationError) as caught:
            TransactionEdge(**self._edge(receiver="alice@upi"))
        assert "same account" in str(caught.value), (
            "the self-transfer refusal no longer explains itself; the message is "
            "what tells the caller why a plausible-looking edge was refused."
        )

    def test_an_empty_batch_is_rejected(self):
        from pydantic import ValidationError

        from api.schemas import ScoringRequest
        with pytest.raises(ValidationError):
            ScoringRequest(transactions=[])

    def test_a_batch_over_the_cap_is_rejected(self):
        from pydantic import ValidationError

        from api.schemas import ScoringRequest
        edges = [self._edge(sender=f"a{i}@upi", receiver=f"b{i}@upi")
                 for i in range(1001)]
        with pytest.raises(ValidationError):
            ScoringRequest(transactions=edges)

    @pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
    def test_an_out_of_range_threshold_override_is_rejected(self, bad):
        """
        Zero is the interesting one. `classify_risk` raises on it because every
        non-negative score satisfies `>= 0`, so a zero threshold rates the entire
        population CRITICAL. Caught here, the caller gets a 422 naming
        `threshold_override`; caught in the responder, they get a 500.
        """
        from pydantic import ValidationError

        from api.schemas import ScoringRequest
        with pytest.raises(ValidationError):
            ScoringRequest(transactions=[self._edge()], threshold_override=bad)

    def test_a_threshold_override_just_above_zero_is_accepted(self):
        """
        The mirror image, so the bound above cannot be satisfied by rejecting
        everything. `gt=0.0` must admit the smallest usable threshold.
        """
        from api.schemas import ScoringRequest
        request = ScoringRequest(transactions=[self._edge()],
                                 threshold_override=1e-6)
        assert request.threshold_override == pytest.approx(1e-6)

    @pytest.mark.parametrize("bad", [0.0, -100.0])
    def test_a_non_positive_amount_is_rejected(self, bad):
        from pydantic import ValidationError

        from api.schemas import TransactionEdge
        with pytest.raises(ValidationError):
            TransactionEdge(**self._edge(amount=bad))

    def test_a_naive_timestamp_is_read_as_utc(self):
        """
        Not cosmetic. A naive datetime compared against the tz-aware context
        column raises `TypeError` inside the window trim, where the traceback
        names pandas internals and not the field that caused it.
        """
        from api.schemas import TransactionEdge
        edge = TransactionEdge(**self._edge(timestamp="2025-04-01T10:00:00"))
        assert edge.timestamp.tzinfo is not None
        assert edge.timestamp.utcoffset().total_seconds() == 0


# ══════════════════════════════════════════════════════════════════
# 10. The endpoint, run for real
# ══════════════════════════════════════════════════════════════════
#
# WHY THIS DOES NOT USE `TestClient`
# ──────────────────────────────────
# requirements.txt excludes httpx on the stated grounds that "no test makes an
# HTTP call; the API is tested by importing it", and starlette's TestClient is an
# httpx client. Rather than add a dependency the repo deliberately declined, these
# tests await the endpoint coroutine directly. That exercises everything this
# repo owns — the merge, the extractor, the booster, TreeSHAP, the responder gate
# — and skips only FastAPI's own request parsing and JSON serialisation, which is
# framework code with its own test suite. Refusals arrive as raised
# `HTTPException`, which is what the framework would have turned into a status
# code anyway.

class TestTheEndpointScoresForReal:

    @pytest.fixture(scope="class")
    def loop(self):
        import asyncio
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="class")
    def service(self, loop):
        """
        The app with its lifespan entered, or a skip explaining what is missing.

        One event loop for the class, because the lifespan is entered once and
        `STATE` must still be populated when the tests run.
        """
        pytest.importorskip("fastapi", reason="api.main imports FastAPI")
        pytest.importorskip("xgboost", reason="api.main imports xgboost")
        pytest.importorskip("networkx", reason="the extractor builds a graph")
        pytest.importorskip("sklearn", reason="api.main imports scikit-learn")

        import api.main as main

        manager = main.lifespan(main.app)
        loop.run_until_complete(manager.__aenter__())
        if not main.STATE.ready:
            reason = main.STATE.degraded_reason or "prerequisites missing"
            loop.run_until_complete(manager.__aexit__(None, None, None))
            pytest.skip(
                f"service cannot score ({reason}). Run `python -m data.generator` "
                f"then `python -m models.train`.")
        yield main
        loop.run_until_complete(manager.__aexit__(None, None, None))

    @staticmethod
    def _edges_inside_the_window(context, n: int = 6) -> list[dict]:
        """
        `n` new transactions between real accounts, dated inside the context span.

        Dated from the context file rather than from a literal, so regenerating the
        data does not silently push every test into the 422 path — which is what a
        hard-coded 2025 date would do, and it would look like a passing test of the
        refusal rather than a broken test of the success path.
        """
        import pandas as pd

        ordered = context.sort_values("timestamp")
        latest = ordered["timestamp"].max()
        accounts = list(dict.fromkeys(
            ordered["sender"].tail(400).tolist()
            + ordered["receiver"].tail(400).tolist()))
        if len(accounts) < 2 * n:
            pytest.skip(f"context graph has only {len(accounts)} accounts")
        return [
            {"sender": accounts[2 * i], "receiver": accounts[2 * i + 1],
             "amount": 25_000.0 + 1_000.0 * i,
             "timestamp": (latest - pd.Timedelta(hours=i + 1)).isoformat()}
            for i in range(n)
        ]

    @pytest.fixture(scope="class")
    def edges(self, service):
        """One batch, built once, deep-copied per use so no test can mutate it."""
        return self._edges_inside_the_window(service.STATE.context_edges)

    @pytest.fixture(scope="class")
    def response(self, service, loop, edges):
        from api.schemas import ScoringRequest
        request = ScoringRequest(transactions=[dict(e) for e in edges])
        return loop.run_until_complete(service.score_transactions(request))

    # ── the success path ──

    def test_it_returns_a_score_for_every_submitted_account(self, response,
                                                            edges):
        submitted = ({e["sender"] for e in edges}
                     | {e["receiver"] for e in edges})
        returned = {ns.node_id for ns in response.node_scores}
        assert submitted <= returned, (
            f"{len(submitted - returned)} submitted account(s) got no score: "
            f"{sorted(submitted - returned)[:5]}"
        )

    def test_the_historical_graph_was_actually_used(self, response):
        """
        The whole point of the merge. Zero context transactions means features
        computed on a handful of edges — PageRank at ~1/n, clustering and cycle
        participation at 0 — and a threshold calibrated on a 60-day graph applied
        to that is well-formed and meaningless.
        """
        assert response.context.n_context_transactions_used > 0
        assert response.context.n_nodes_in_graph > len(response.node_scores), (
            "the merged graph is no larger than the returned batch, so no history "
            "was merged in."
        )

    def test_the_effective_window_matches_the_trained_one(self, response):
        assert response.context.window_comparable, (
            f"observation window {response.context.observation_window_days}d "
            f"against a trained {response.context.trained_window_days}d — "
            f"magnitude features are on a different scale than in training. "
            f"Warnings: {response.context.warnings}"
        )

    def test_every_action_follows_from_its_own_score(self, response, service):
        """
        The gate's invariant, checked on real scores rather than on a forgery.

        tests/test_responder.py proves `validate_response_batch` rejects a
        downgraded action. This proves the batch that actually leaves the endpoint
        satisfies the thing it was checked against — recomputed here from the
        threshold the response itself reports, so a response cannot pass by
        agreeing with a threshold nobody applied.
        """
        from api.responder import classify_risk, determine_action

        break_even = service.STATE.break_even_probability
        for ns in response.node_scores:
            expected_level = classify_risk(
                ns.risk_score, response.threshold_used, break_even)
            assert ns.risk_level == expected_level, (
                f"{ns.node_id}: score {ns.risk_score} at threshold "
                f"{response.threshold_used} is {expected_level.value}, but the "
                f"response says {ns.risk_level.value}."
            )
            assert ns.action == determine_action(expected_level), (
                f"{ns.node_id}: action {ns.action.value} does not follow from "
                f"{expected_level.value}."
            )

    def test_no_response_carries_an_enforcement_action(self, response):
        """
        The action ceiling, asserted against a set written out here rather than
        imported from api/responder.py. A permitted-set read from the module under
        test can only describe what that module already believes.
        """
        permitted = {"ALLOW", "REQUIRE_ADDITIONAL_AUTH", "HOLD_FOR_REVIEW"}
        actual = {ns.action.value for ns in response.node_scores}
        assert actual <= permitted, (
            f"the endpoint returned action(s) {sorted(actual - permitted)}, "
            f"outside the defense-only ceiling of HOLD_FOR_REVIEW."
        )

    def test_the_same_batch_scores_identically_twice(self, service, loop, edges):
        """
        The frozen reference partition, observable from outside.

        Louvain shuffles node visit order from its seed, so a per-request
        repartition moved `community_internal_ratio` between two identical
        requests — v2's defect, invisible because both answers looked reasonable.
        """
        from api.schemas import ScoringRequest

        first, second = (
            loop.run_until_complete(service.score_transactions(
                ScoringRequest(transactions=[dict(e) for e in edges])))
            for _ in range(2))
        by_node_first = {ns.node_id: ns.risk_score for ns in first.node_scores}
        by_node_second = {ns.node_id: ns.risk_score for ns in second.node_scores}
        assert by_node_first == by_node_second, (
            "two identical requests produced different scores; the partition is "
            "not frozen, so no alert in this system is reproducible."
        )
        assert first.context.partition_fingerprint == \
            second.context.partition_fingerprint

    # ── explanations ──

    def test_explanations_are_per_account_or_withheld_with_a_reason(
            self, response):
        """
        v2 filled every alert with the model's three highest global gains, so the
        explanation was identical on every account in every batch and described the
        model rather than the case.

        The discriminator is the CONTRIBUTIONS, not the feature names. Two accounts
        with similar behaviour can legitimately share the same top three features,
        so a name-based check would fail on correct output. A global importance
        list, by contrast, is a property of the booster: it cannot vary between two
        accounts in the same batch. Signed log-odds contributions can, and must.

        The alternative permitted outcome is no factors at all together with a
        warning saying so. A silent empty list is the one thing that is not
        allowed: an alert with no reason is recoverable, an alert whose missing
        reason is unexplained is not diagnosable.
        """
        signatures = {tuple(round(f.contribution, 9)
                            for f in ns.contributing_factors)
                      for ns in response.node_scores}
        if signatures == {()}:
            joined = " ".join(response.context.warnings).lower()
            assert "attribution" in joined or "shap" in joined, (
                "no account received contributing factors and no warning explains "
                "why."
            )
            return
        assert len(signatures) > 1, (
            f"every account in the batch was explained by the identical set of "
            f"contributions {signatures.pop()}. Contributions are per-account by "
            f"construction; identical ones across a batch are what a global "
            f"feature-importance list looks like."
        )

    def test_each_factor_declares_the_direction_its_number_shows(self, response):
        """
        `effect` is what an analyst reads; `contribution` is what the model
        computed. If they can disagree, the readable half is decoration.
        """
        for ns in response.node_scores:
            for f in ns.contributing_factors:
                expected = "raises_risk" if f.contribution > 0 else "lowers_risk"
                assert f.effect == expected, (
                    f"{ns.node_id}: {f.feature} contributes "
                    f"{f.contribution:+.4f} log-odds but is labelled "
                    f"{f.effect!r}."
                )

    def test_every_named_factor_is_a_real_contract_feature(self, response):
        from models.features import FEATURE_COLS

        for ns in response.node_scores:
            for factor in ns.contributing_factors:
                assert factor.feature in FEATURE_COLS, (
                    f"{ns.node_id} was explained by {factor.feature!r}, which is "
                    f"not in the feature contract. An analyst cannot act on a "
                    f"name the model does not use."
                )
                assert factor.effect in ("raises_risk", "lowers_risk")

    # ── provenance ──

    def test_an_override_is_applied_and_declared(self, service, loop, edges):
        from api.schemas import ScoringRequest

        response = loop.run_until_complete(service.score_transactions(
            ScoringRequest(transactions=[dict(e) for e in edges],
                           threshold_override=0.9)))
        assert response.threshold_used == pytest.approx(0.9)
        assert response.context.threshold_source == "request_override", (
            "the response does not record that the threshold came from the "
            "request, so an alert cannot be told apart from one raised at the "
            "shipped operating point."
        )

    def test_the_default_threshold_is_the_published_one(self, response, service):
        """
        The other half of the override test. A response that silently scored at
        some other threshold would still be internally consistent — every action
        would follow from the score — and would still be wrong, because the
        reported cost figures hold at exactly one operating point.
        """
        assert response.context.threshold_source == "metrics.json"
        assert response.threshold_used == pytest.approx(service.STATE.threshold), (
            f"the batch was scored at {response.threshold_used} while "
            f"metrics.json publishes {service.STATE.threshold}."
        )

    # ── refusals ──

    def test_a_batch_dated_outside_the_context_is_refused(self, service, loop,
                                                          edges):
        """
        The 422. This is the common case, not the exotic one: any present-day
        timestamp lands here, because the shipped context file ends in 2025.
        """
        import pandas as pd
        from fastapi import HTTPException

        from api.schemas import ScoringRequest

        far_future = (service.STATE.context_edges["timestamp"].max()
                      + pd.Timedelta(days=3_650))
        stale = [dict(e, timestamp=(far_future + pd.Timedelta(hours=i)).isoformat())
                 for i, e in enumerate(edges)]

        with pytest.raises(HTTPException) as caught:
            loop.run_until_complete(service.score_transactions(
                ScoringRequest(transactions=stale)))
        assert caught.value.status_code == 422, (
            f"expected a 422 for a batch with no surviving context, got "
            f"{caught.value.status_code}."
        )
        assert "context" in str(caught.value.detail).lower(), (
            "the refusal does not tell the caller that the problem is the dates; "
            "without that they cannot fix the request."
        )

    def test_an_unready_service_refuses_with_503(self, service, loop, edges,
                                                 monkeypatch):
        from fastapi import HTTPException

        from api.schemas import ScoringRequest

        monkeypatch.setattr(service.STATE, "model", None)
        with pytest.raises(HTTPException) as caught:
            loop.run_until_complete(service.score_transactions(
                ScoringRequest(transactions=[dict(e) for e in edges])))
        assert caught.value.status_code == 503

    # ── /health ──

    def test_health_publishes_the_threshold_it_is_scoring_at(self, service,
                                                              loop):
        health = loop.run_until_complete(service.health_check())
        assert health.status == "healthy"
        assert health.optimal_threshold is not None
        assert health.partition_fingerprint, (
            "/health publishes no partition fingerprint, so two replicas that "
            "partitioned differently would be indistinguishable."
        )
        assert health.feature_contract_verified

    def test_health_reports_a_missing_threshold_as_missing(self, service, loop,
                                                           monkeypatch):
        """
        The regression test for a real defect: `HealthResponse.optimal_threshold`
        was `float = 0.5` while api/main.py passes `STATE.threshold`, which is
        `float | None` and is None whenever the threshold could not be read. Under
        pydantic v2 that raised a ValidationError, so /health returned a 500 in
        precisely the degraded state /health exists to describe — and the comment
        beside the call already claimed the field was `float | None`.
        """
        monkeypatch.setattr(service.STATE, "threshold", None)
        monkeypatch.setattr(service.STATE, "threshold_source", "none")
        health = loop.run_until_complete(service.health_check())
        assert health.status == "degraded"
        assert health.optimal_threshold is None, (
            f"/health reported {health.optimal_threshold!r} as the operating "
            f"threshold while the service has none. Any concrete number here is "
            f"read as a real operating point."
        )
