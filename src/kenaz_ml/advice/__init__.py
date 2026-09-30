"""Harness recommendation kinds — contracts, dispatch, and label ingest.

The dispatch layer the harness's ``SidecarAdvisor`` calls through
``/v1/recommend/{kind}`` and pushes labels to through ``/v1/labels/{kind}``
(two-client-engine-01MSK2EN). Kept free of eager imports: each submodule is
imported where it is used.
"""
