"""Feature contracts for the harness recommendation kinds (D-A4).

One ordered :class:`~kenaz_ml.modelstore.registry.FeatureContract` per kind —
``branch_now``, ``compact_now``, ``escalate_model`` — shared by the dispatch
layer (``/v1/recommend``, ``/v1/contracts``), label ingest (``/v1/labels``) and,
later, training (``harness-recommendation-models-01MSK2RM``). No other module
keeps its own copy: look contracts up through :func:`contract_for`.

Source of truth
---------------
These features are computed **harness-side** and arrive already computed in
the request body, so there is no daemon-observed event stream for Feast to materialize
and these contracts are *not* Feast-derived. The authoritative feature lists
are ``typewriter/spec.yaml`` in this repository: the ordered names, the
``ERROR_KINDS`` enum and ``VOCABULARY_VERSION`` are generated from it into
:mod:`kenaz_ml.typewriter.vocab` (re-exported below), and the same spec
generates the Go module the harness builds its feature structs from. A name
change or reorder moves ``service_version`` automatically; a change to what a
feature *means* requires incrementing ``vocabulary_version`` in the spec.

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

from kenaz_ml.typewriter.vocab import ERROR_KINDS, FEATURE_DTYPE, KIND_FEATURES, VOCABULARY_VERSION

#: ``VOCABULARY_VERSION`` is the semantic salt folded into every contract's
#: ``service_version``; ``ERROR_KINDS`` is the harness's closed ``ErrorCategory``
#: enum, whose one-hots ``escalate_model`` expands in enum order. Both, with
#: ``FEATURE_DTYPE`` and the per-kind ordered names in ``KIND_FEATURES``, are
#: generated from ``typewriter/spec.yaml`` and re-exported here.
BRANCH_NOW_FEATURES: tuple[str, ...] = KIND_FEATURES["branch_now"]
COMPACT_NOW_FEATURES: tuple[str, ...] = KIND_FEATURES["compact_now"]
ESCALATE_MODEL_FEATURES: tuple[str, ...] = KIND_FEATURES["escalate_model"]

__all__ = [
    "BRANCH_NOW_FEATURES",
    "COMPACT_NOW_FEATURES",
    "ERROR_KINDS",
    "ESCALATE_MODEL_FEATURES",
    "FEATURE_DTYPE",
    "KIND_FEATURES",
    "KIND_IDS",
    "VOCABULARY_VERSION",
    "all_contracts",
    "build_contract",
    "contract_for",
    "contract_version",
]

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
