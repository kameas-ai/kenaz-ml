"""Laya serving integration (``laya-serving-and-packs-01MSK2SP``).

Everything laya-specific lives here, and **nothing here imports laya, torch or
onnxruntime at module import time** (cold-start budget; and because laya is not
a declared dependency -- see ``agent.py`` for the distribution blocker). Each
heavy import happens inside the function that needs it.

Modules:

* :mod:`kenaz_ml.laya.systemone_mount` -- the raw ``/v1/systemone`` (+ ``/batch``)
  pass-through (WP01).
* :mod:`kenaz_ml.laya.agent` -- the in-process ``ONNXAgent`` wrapper (WP02).
* :mod:`kenaz_ml.laya.eligibility` -- the per-host eligibility gate (WP03).
"""
