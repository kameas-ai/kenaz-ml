"""Hand-authored feature contracts for the harness recommendation kinds (D-A4).

One ordered :class:`~kenaz_ml.modelstore.registry.FeatureContract` per kind —
``branch_now``, ``compact_now``, ``escalate_model`` — shared by the dispatch
layer (``/v1/recommend``, ``/v1/contracts``), label ingest (``/v1/labels``) and,
later, training (``harness-recommendation-models-01MSK2RM``). No other module
keeps its own copy: look contracts up through :func:`contract_for`.

Source of truth, and the sync obligation
----------------------------------------
These features are computed **harness-side** and arrive already computed in
the request body, so there is no daemon-observed event stream for Feast to materialize
and these contracts are *not* Feast-derived. The authoritative feature lists
are the kenaz-harness design doc's §4 recommendation catalog
(``kitty-specs/laya-advisors-01LAYA001/research/kenaz-ml-integration-design.md``)
and the harness kind registry that implements it. Nothing in this repository
can check that the harness still posts these names: **a harness-side contract
change requires a matching change here and a version bump** (a name change or
reorder moves ``service_version`` automatically; a change to what a feature
*means* requires incrementing :data:`VOCABULARY_VERSION`).

If a Feast feature service is later registered for any of these kinds, the
constant here becomes that service's seed — not a competing source of truth.

``service_version`` recipe
--------------------------
Mirrors :func:`kenaz_ml.feature_store.materialize.feature_service_version`
exactly — a 16-hex sha256 over ``"|".join([service, *"service:feature"...])`` —
with the semantic salt appended as one final ``"vocabulary:<N>"`` element::

    sha256("|".join([kind, *(f"{kind}:{name}" for name in names),
                     f"vocabulary:{VOCABULARY_VERSION}"]))[:16]

So the version moves when a feature is added, removed, renamed or reordered,
and when :data:`VOCABULARY_VERSION` is incremented. It is never empty, which
``retained.append_examples`` requires before it will stamp a retained set.
:data:`VOCABULARY_VERSION` is this family's own constant;
``feature-vocabulary-refresh-01MSK2VR`` keeps a separate one for the
Feast-derived workbench contracts, so a semantics change in one family never
resets the other's retained data.

Dtypes are ``"float64"`` for every feature — the registry's convention
(``str(Float64).lower()``) — because the harness posts JSON numbers.

Only the standard library and project-internal modules are imported (NFR-001).
The registry import is deferred to first use.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from kenaz_ml.modelstore.registry import FeatureContract

#: Semantic salt folded into every contract's ``service_version``. Increment it
#: when what any feature *means* changes (units, window, extraction rule) while
#: its name and position stay the same — the one kind of change a name/order
#: hash cannot see. Initial value 1 (D-A4, extended 2026-09-30).
VOCABULARY_VERSION = 1

#: The dtype every harness-kind feature is declared with (registry convention).
FEATURE_DTYPE = "float64"

#: The harness's closed ``ErrorCategory`` enum, in order — a **cross-repo
#: constant** (kenaz-harness ``core/fleet/usage_emitter.go``, double-projected on
#: the chat path, so nothing outside this set can arrive; a category unknown at
#: the wire buckets to ``unknown``). A harness-side change to the enum requires a
#: matching change here and a :data:`VOCABULARY_VERSION` bump. Ruled 2026-09-30.
ERROR_KINDS: tuple[str, ...] = ("auth", "transient", "cancelled", "budget", "unknown")

#: branch_now — design doc §4. The heuristic counts come from the shipped
#: ``core/branchadvisor`` detector, demoted to a feature extractor harness-side.
BRANCH_NOW_FEATURES: tuple[str, ...] = (
    "turns_since_session_start",
    "turns_since_last_branch",
    "prior_branch_count",
    "edit_resend_precursor",
    "heuristic_signal_count",
    "heuristic_noise_count",
    "last_user_msg_len",
    "tool_call_density_window",
)

#: compact_now — design doc §4; every field is present on the harness's
#: ``SessionCompactedPayload``.
COMPACT_NOW_FEATURES: tuple[str, ...] = (
    "context_fill_fraction",
    "tokens_in_span",
    "turns_since_last_compaction",
    "tool_result_token_fraction",
    "model_context_limit",
    "historical_compression_ratio",
)

#: escalate_model — design doc §4, with the ``error_kind`` one-hots expanded over
#: :data:`ERROR_KINDS` in enum order, at the catalog's position.
ESCALATE_MODEL_FEATURES: tuple[str, ...] = (
    "consecutive_tool_failures",
    "retries_in_window",
    "turn_latency_trend",
    "current_rung",
    *(f"error_kind_{kind}" for kind in ERROR_KINDS),
    "budget_remaining_fraction",
)

#: Kind id -> ordered feature names. Insertion order is the published order.
KIND_FEATURES: dict[str, tuple[str, ...]] = {
    "branch_now": BRANCH_NOW_FEATURES,
    "compact_now": COMPACT_NOW_FEATURES,
    "escalate_model": ESCALATE_MODEL_FEATURES,
}

#: The harness kind ids this engine registers, in publication order.
KIND_IDS: tuple[str, ...] = tuple(KIND_FEATURES)


def contract_version(kind_id: str, names: tuple[str, ...], vocabulary_version: int = VOCABULARY_VERSION) -> str:
    """The deterministic, non-empty ``service_version`` for a hand-authored contract.

    See the module docstring for the recipe. Exposed with explicit inputs so a
    test can prove the hash covers order and the salt.
    """
    references = [f"{kind_id}:{name}" for name in names]
    payload = "|".join([kind_id, *references, f"vocabulary:{vocabulary_version}"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_contract(
    kind_id: str, names: tuple[str, ...], vocabulary_version: int = VOCABULARY_VERSION
) -> FeatureContract:
    """Build the ``FeatureContract`` for ``kind_id`` over ``names`` (in order)."""
    from kenaz_ml.modelstore.registry import FeatureContract

    ordered = tuple(names)
    return FeatureContract(
        service=kind_id,
        service_version=contract_version(kind_id, ordered, vocabulary_version),
        names=ordered,
        dtypes=tuple(FEATURE_DTYPE for _ in ordered),
        extra={"source": "hand_authored", "vocabulary_version": vocabulary_version},
    )


_CONTRACTS: dict[str, FeatureContract] = {}


def contract_for(kind_id: str) -> FeatureContract | None:
    """The published contract for ``kind_id``, or ``None`` for an unknown kind."""
    names = KIND_FEATURES.get(kind_id)
    if names is None:
        return None
    cached = _CONTRACTS.get(kind_id)
    if cached is None:
        cached = build_contract(kind_id, names)
        _CONTRACTS[kind_id] = cached
    return cached


def all_contracts() -> dict[str, FeatureContract]:
    """Every registered harness kind's contract, in publication order."""
    out: dict[str, FeatureContract] = {}
    for kind_id in KIND_IDS:
        contract = contract_for(kind_id)
        if contract is not None:
            out[kind_id] = contract
    return out
