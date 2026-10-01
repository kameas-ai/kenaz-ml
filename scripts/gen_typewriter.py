#!/usr/bin/env python3
"""Generate the Python and Go wire types from typewriter/spec.yaml.

Usage:
    python scripts/gen_typewriter.py                    # write every generated file
    python scripts/gen_typewriter.py --check            # exit 1 if any is out of date
    python scripts/gen_typewriter.py --check --strict   # ... and fail if gofmt is missing

The spec is the source of truth; see typewriter/README.md for the dialect.
Python output is formatted with ``ruff format`` and Go output with ``gofmt``, so
the checked-in files are exactly what the formatters would leave behind. Without
``gofmt`` on PATH the Go file is skipped (``--strict`` makes that an error).

This script deliberately imports nothing from ``kenaz_ml``: the package imports
the files generated here, so the generator must run when they are absent.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = ROOT / "typewriter" / "spec.yaml"
PY_VOCAB_PATH = ROOT / "src" / "kenaz_ml" / "typewriter" / "vocab.py"
PY_MODELS_PATH = ROOT / "src" / "kenaz_ml" / "typewriter" / "models.py"
GO_PATH = ROOT / "typewriter" / "typewriter.gen.go"

SPEC_VERSION = 1
PRIMITIVES = ("string", "int", "float", "bool", "any", "object")
CONSTRAINTS = ("min_length", "gt", "ge", "le")
FIELD_KEYS = {"type", "nullable", "default", "description", "comment", "vocabulary", *CONSTRAINTS}
GO_INITIALISMS = {"id": "ID", "pid": "PID", "ts": "TS", "ms": "MS", "sha256": "SHA256", "sha8": "SHA8", "io": "IO"}


class SpecError(ValueError):
    """The spec is not well-formed."""


# ---------------------------------------------------------------------------
# Spec model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypeExpr:
    """A parsed type expression: a primitive, a named type, list<T>, map<T> or a union."""

    kind: str  # "prim" | "named" | "list" | "map" | "union"
    name: str = ""
    args: tuple[TypeExpr, ...] = ()


@dataclass(frozen=True)
class FieldSpec:
    name: str
    type: TypeExpr
    nullable: bool
    has_default: bool
    default: Any
    description: str | None
    comment: str | None
    vocabulary: str | None
    constraints: dict[str, Any]


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    description: str | None
    fields: tuple[FieldSpec, ...]
    exactly_one_of: tuple[str, ...]


@dataclass(frozen=True)
class EnumSpec:
    name: str
    description: str
    values: tuple[str, ...]


@dataclass(frozen=True)
class KindSpec:
    kind_id: str
    description: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class Spec:
    go_module: str
    vocabulary_version: int
    feature_dtype: str
    enums: tuple[EnumSpec, ...]
    kinds: tuple[KindSpec, ...]
    schemas: tuple[SchemaSpec, ...]

    def enum(self, name: str) -> EnumSpec | None:
        return next((e for e in self.enums if e.name == name), None)


def parse_type(text: str) -> TypeExpr:
    text = text.strip()
    if "|" in _strip_brackets(text):
        return TypeExpr("union", args=tuple(parse_type(part) for part in _split_top(text, "|")))
    for wrapper in ("list", "map"):
        if text.startswith(f"{wrapper}<") and text.endswith(">"):
            return TypeExpr(wrapper, args=(parse_type(text[len(wrapper) + 1 : -1]),))
    if text in PRIMITIVES:
        return TypeExpr("prim", name=text)
    if re.fullmatch(r"[A-Z][A-Za-z0-9]*", text):
        return TypeExpr("named", name=text)
    raise SpecError(f"unparseable type {text!r}")


def _strip_brackets(text: str) -> str:
    depth, out = 0, []
    for ch in text:
        depth += ch == "<"
        if depth == 0:
            out.append(ch)
        depth -= ch == ">"
    return "".join(out)


def _split_top(text: str, sep: str) -> list[str]:
    parts, depth, current = [], 0, []
    for ch in text:
        depth += ch == "<"
        depth -= ch == ">"
        if ch == sep and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    parts.append("".join(current))
    return parts


def _named_types(expr: TypeExpr) -> list[str]:
    if expr.kind == "named":
        return [expr.name]
    return [name for arg in expr.args for name in _named_types(arg)]


def load_spec(path: Path = SPEC_PATH) -> Spec:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("typewriter") != SPEC_VERSION:
        raise SpecError(f"spec dialect must be {SPEC_VERSION}, got {raw.get('typewriter')!r}")

    enums = tuple(
        EnumSpec(name, str(body["description"]).strip(), tuple(body["values"])) for name, body in raw["enums"].items()
    )
    enum_names = {e.name for e in enums}
    for e in enums:
        if len(set(e.values)) != len(e.values) or not e.values:
            raise SpecError(f"enum {e.name}: values must be non-empty and unique")

    kinds = []
    for kind_id, body in raw["kinds"].items():
        names: list[str] = []
        for item in body["features"]:
            if isinstance(item, str):
                names.append(item)
                continue
            source = next((e for e in enums if e.name == item["one_hot"]), None)
            if source is None:
                raise SpecError(f"kind {kind_id}: one_hot names unknown enum {item['one_hot']!r}")
            names.extend(f"{item['prefix']}{value}" for value in source.values)
        if len(set(names)) != len(names):
            raise SpecError(f"kind {kind_id}: duplicate feature name")
        kinds.append(KindSpec(kind_id, str(body["description"]).strip(), tuple(names)))

    schemas = []
    seen: set[str] = set()
    for name, body in raw["schemas"].items():
        fields = []
        for field_name, field_body in body["fields"].items():
            if isinstance(field_body, str):
                field_body = {"type": field_body}
            unknown = set(field_body) - FIELD_KEYS
            if unknown:
                raise SpecError(f"{name}.{field_name}: unknown keys {sorted(unknown)}")
            expr = parse_type(field_body["type"])
            for ref in _named_types(expr):
                if ref not in enum_names and ref not in seen:
                    raise SpecError(f"{name}.{field_name}: {ref!r} is not an enum or an earlier schema")
            vocabulary = field_body.get("vocabulary")
            if vocabulary is not None and vocabulary not in enum_names:
                raise SpecError(f"{name}.{field_name}: vocabulary {vocabulary!r} is not an enum")
            nullable = bool(field_body.get("nullable", False))
            has_default = "default" in field_body
            if has_default and field_body["default"] is None and not nullable and expr != TypeExpr("prim", "any"):
                raise SpecError(f"{name}.{field_name}: a null default needs nullable: true")
            fields.append(
                FieldSpec(
                    name=field_name,
                    type=expr,
                    nullable=nullable,
                    has_default=has_default,
                    default=field_body.get("default"),
                    description=_clean(field_body.get("description")),
                    comment=_clean(field_body.get("comment")),
                    vocabulary=vocabulary,
                    constraints={k: field_body[k] for k in CONSTRAINTS if k in field_body},
                )
            )
        one_of = tuple(body.get("exactly_one_of", ()))
        if set(one_of) - {f.name for f in fields}:
            raise SpecError(f"{name}: exactly_one_of names an unknown field")
        schemas.append(SchemaSpec(name, _clean(body.get("description")), tuple(fields), one_of))
        seen.add(name)

    return Spec(
        go_module=raw["go_module"],
        vocabulary_version=int(raw["vocabulary_version"]),
        feature_dtype=str(raw["feature_dtype"]),
        enums=enums,
        kinds=tuple(kinds),
        schemas=tuple(schemas),
    )


def _clean(text: Any) -> str | None:
    return None if text is None else str(text).strip("\n").rstrip()


def contract_version(kind_id: str, names: tuple[str, ...], vocabulary_version: int) -> str:
    """The kind's 16-hex contract version. Must equal ``kenaz_ml.advice.contracts.contract_version``
    (``tests/test_typewriter.py`` pins the two together)."""
    payload = "|".join([kind_id, *(f"{kind_id}:{name}" for name in names), f"vocabulary:{vocabulary_version}"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def upper_snake(camel: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", camel).upper()


def plural_constant(enum_name: str) -> str:
    return f"{upper_snake(enum_name)}S"


def go_name(snake: str) -> str:
    return "".join(GO_INITIALISMS.get(part, part[:1].upper() + part[1:]) for part in snake.split("_"))


def features_class(kind_id: str) -> str:
    return f"{go_name(kind_id)}Features"


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

PY_HEADER = '"""{summary}\n\nGenerated by scripts/gen_typewriter.py from typewriter/spec.yaml. DO NOT EDIT:\nchange the spec and run ``make typewriter``.\n"""\n\nfrom __future__ import annotations\n'


def _py_tuple(values: tuple[str, ...]) -> str:
    return "(" + "".join(f"{v!r}, " for v in values) + ")"


def render_python_vocab(spec: Spec) -> str:
    out = [
        PY_HEADER.format(summary="The engine's shared vocabularies: kinds, ordered features and closed enums."),
        "# Standard library only: kenaz_ml.advice.contracts imports this module (NFR-001).\n",
        f"VOCABULARY_VERSION = {spec.vocabulary_version}\n",
        f"FEATURE_DTYPE = {spec.feature_dtype!r}\n",
    ]
    for enum in spec.enums:
        out.append(_py_comment(enum.description))
        out.append(f"{plural_constant(enum.name)}: tuple[str, ...] = {_py_tuple(enum.values)}\n")
    out.append("# Kind id -> ordered feature names. The order is the vector layout.")
    out.append("KIND_FEATURES: dict[str, tuple[str, ...]] = {")
    for kind in spec.kinds:
        out.append(f"    {kind.kind_id!r}: {_py_tuple(kind.features)},")
    out.append("}\n")
    out.append("# Kind id -> the 16-hex contract version a client sends as feature_contract_version.")
    out.append("KIND_CONTRACT_VERSIONS: dict[str, str] = {")
    for kind in spec.kinds:
        version = contract_version(kind.kind_id, kind.features, spec.vocabulary_version)
        out.append(f"    {kind.kind_id!r}: {version!r},")
    out.append("}\n")
    return "\n".join(out)


def _py_comment(text: str, indent: str = "") -> str:
    lines = textwrap.wrap(text, width=100 - len(indent)) or [""]
    return "\n".join(f"{indent}# {line}" for line in lines)


def py_type(spec: Spec, expr: TypeExpr) -> str:
    if expr.kind == "prim":
        return {
            "string": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "any": "Any",
            "object": "dict[str, Any]",
        }[expr.name]
    if expr.kind == "named":
        enum = spec.enum(expr.name)
        if enum is not None:
            return "Literal[" + ", ".join(f'"{v}"' for v in enum.values) + "]"
        return expr.name
    if expr.kind == "list":
        return f"list[{py_type(spec, expr.args[0])}]"
    if expr.kind == "map":
        return f"dict[str, {py_type(spec, expr.args[0])}]"
    return " | ".join(py_type(spec, arg) for arg in expr.args)


def _py_docstring(text: str) -> str:
    if '"""' in text or "\\" in text or text.endswith('"'):
        raise SpecError(f"description cannot be rendered as a docstring: {text[:40]!r}")
    lines = text.split("\n")
    if len(lines) == 1:
        return f'    """{lines[0]}"""'
    body = "\n".join(f"    {line}".rstrip() for line in lines[1:])
    return f'    """{lines[0]}\n{body}\n    """'


def _py_field(spec: Spec, field: FieldSpec) -> str:
    annotation = py_type(spec, field.type)
    if field.nullable:
        annotation += " | None"
    args: list[str] = []
    if not field.has_default:
        args.append("...")
    elif field.default == {}:
        args.append("default_factory=dict")
    elif field.default == []:
        args.append("default_factory=list")
    else:
        args.append(repr(field.default))
    for key, value in field.constraints.items():
        args.append(f"{key}={value!r}")
    if field.description is not None:
        args.append(f"description={field.description!r}")
    if len(args) > 1:
        line = f"    {field.name}: {annotation} = Field({', '.join(args)})"
    elif field.has_default and not args[0].startswith("default_factory"):
        line = f"    {field.name}: {annotation} = {args[0]}"
    elif field.has_default:
        line = f"    {field.name}: {annotation} = Field({args[0]})"
    else:
        line = f"    {field.name}: {annotation}"
    if field.comment is not None:
        line = _py_comment(field.comment, "    ") + "\n" + line
    return line


def render_python_models(spec: Spec) -> str:
    out = [
        PY_HEADER.format(summary="The engine's wire models and per-kind feature models."),
        "from typing import Any, Literal\n",
        "from pydantic import BaseModel, ConfigDict, Field, model_validator\n\n",
    ]
    for schema in spec.schemas:
        out.append(f"class {schema.name}(BaseModel):")
        if schema.description:
            out.append(_py_docstring(schema.description) + "\n")
        out.extend(_py_field(spec, field) for field in schema.fields)
        if schema.exactly_one_of:
            names = schema.exactly_one_of
            count = " + ".join(f"(self.{n} is not None)" for n in names)
            out.append("")
            out.append('    @model_validator(mode="after")')
            out.append(f"    def _exactly_one_answer(self) -> {schema.name}:")
            out.append(f"        if {count} != 1:")
            out.append(f'            raise ValueError("exactly one of {" or ".join(names)} must be set")')
            out.append("        return self")
        out.append("\n")
    for kind in spec.kinds:
        out.append(f"class {features_class(kind.kind_id)}(BaseModel):")
        out.append(
            f'    """The ``{kind.kind_id}`` feature vector, in contract order. {kind.description}\n\n'
            "    The wire carries these as a name -> number object (``RecommendRequest.features``);\n"
            "    this model is the typed form of that object.\n"
            '    """\n'
        )
        out.append('    model_config = ConfigDict(extra="forbid")\n')
        out.extend(f"    {name}: float" for name in kind.features)
        out.append("\n")
    out.append("# Kind id -> its feature model.")
    out.append("KIND_FEATURE_MODELS: dict[str, type[BaseModel]] = {")
    out.extend(f"    {kind.kind_id!r}: {features_class(kind.kind_id)}," for kind in spec.kinds)
    out.append("}\n")
    out.append("# Every wire schema, by name.")
    out.append("SCHEMAS: dict[str, type[BaseModel]] = {")
    out.extend(f"    {schema.name!r}: {schema.name}," for schema in spec.schemas)
    out.extend(f"    {features_class(kind.kind_id)!r}: {features_class(kind.kind_id)}," for kind in spec.kinds)
    out.append("}\n")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------


def _go_comment(text: str, indent: str = "") -> list[str]:
    out: list[str] = []
    for paragraph in text.replace("``", "`").split("\n\n"):
        if out:
            out.append(f"{indent}//")
        if any(line.lstrip().startswith("* ") for line in paragraph.split("\n")):
            out.extend(f"{indent}// {line}".rstrip() for line in paragraph.split("\n"))
        else:
            out.extend(f"{indent}// {line}" for line in textwrap.wrap(" ".join(paragraph.split()), 76 - len(indent)))
    return out


def go_type(spec: Spec, expr: TypeExpr) -> str:
    if expr.kind == "prim":
        return {
            "string": "string",
            "int": "int64",
            "float": "float64",
            "bool": "bool",
            "any": "any",
            "object": "map[string]any",
        }[expr.name]
    if expr.kind == "named":
        return expr.name
    if expr.kind == "list":
        return f"[]{go_type(spec, expr.args[0])}"
    if expr.kind == "map":
        return f"map[string]{go_type(spec, expr.args[0])}"
    return "json.RawMessage"


def _go_field(spec: Spec, field: FieldSpec) -> list[str]:
    base = go_type(spec, field.type)
    pointer = field.nullable and not base.startswith(("[]", "map[")) and base not in ("any", "json.RawMessage")
    # omitempty only where the zero value means "absent": a scalar with a
    # non-null default is always sent, so a legitimate 0 or "" is never dropped.
    nilable = field.nullable or base.startswith(("[]", "map[")) or base in ("any", "json.RawMessage")
    tag = field.name + (",omitempty" if field.has_default and nilable else "")
    notes = [text for text in (field.description, field.comment) if text]
    if field.vocabulary:
        notes.append(f"One of the {field.vocabulary} values.")
    out: list[str] = []
    for index, note in enumerate(notes):
        if index:
            out.append("\t//")
        out.extend(_go_comment(note, "\t"))
    out.append(f'\t{go_name(field.name)} {"*" if pointer else ""}{base} `json:"{tag}"`')
    return out


def render_go(spec: Spec) -> str:
    out = [
        "// Code generated by scripts/gen_typewriter.py from spec.yaml. DO NOT EDIT.",
        "",
        "// Package typewriter is the kenaz-ml engine's wire contract: the request and",
        "// response shapes, the recommendation kinds with their ordered feature",
        "// vectors, and the closed vocabularies. It is generated from spec.yaml in",
        "// the kenaz-ml repository, the same file the engine's own models come from.",
        "package typewriter",
        "",
        'import "encoding/json"',
        "",
        "// VocabularyVersion is the semantic salt folded into every contract version.",
        f"const VocabularyVersion = {spec.vocabulary_version}",
        "",
        "// FeatureDtype is the dtype every kind feature is declared with.",
        f'const FeatureDtype = "{spec.feature_dtype}"',
        "",
    ]
    for enum in spec.enums:
        out.extend(_go_comment(f"{enum.name}: {enum.description}"))
        out.append(f"type {enum.name} string")
        out.append("")
        out.append("const (")
        out.extend(f'\t{enum.name}{go_name(value)} {enum.name} = "{value}"' for value in enum.values)
        out.append(")")
        out.append("")
        out.append(f"// {enum.name}s lists every {enum.name}, in spec order.")
        out.append(f"var {enum.name}s = []{enum.name}{{")
        out.extend(f"\t{enum.name}{go_name(value)}," for value in enum.values)
        out.append("}")
        out.append("")

    out.append("// Recommendation kind ids.")
    out.append("const (")
    out.extend(f'\tKind{go_name(kind.kind_id)} = "{kind.kind_id}"' for kind in spec.kinds)
    out.append(")")
    out.append("")
    out.append("// KindIDs lists every kind, in publication order.")
    out.append("var KindIDs = []string{" + ", ".join(f"Kind{go_name(k.kind_id)}" for k in spec.kinds) + "}")
    out.append("")
    out.append("// ContractVersions maps a kind to the 16-hex contract version this build")
    out.append("// was generated against: the value to send as feature_contract_version.")
    out.append("var ContractVersions = map[string]string{")
    out.extend(f"\tKind{go_name(k.kind_id)}: {go_name(k.kind_id)}ContractVersion," for k in spec.kinds)
    out.append("}")
    out.append("")
    out.append("// FeatureNames maps a kind to its ordered feature names (the vector layout).")
    out.append("var FeatureNames = map[string][]string{")
    out.extend(f"\tKind{go_name(k.kind_id)}: {go_name(k.kind_id)}FeatureNames," for k in spec.kinds)
    out.append("}")
    out.append("")

    for kind in spec.kinds:
        camel = go_name(kind.kind_id)
        struct = features_class(kind.kind_id)
        version = contract_version(kind.kind_id, kind.features, spec.vocabulary_version)
        out.append(f"// {camel}ContractVersion is the {kind.kind_id} contract version.")
        out.append(f'const {camel}ContractVersion = "{version}"')
        out.append("")
        out.append(f"// {camel}FeatureNames is the {kind.kind_id} vector layout.")
        out.append(f"var {camel}FeatureNames = []string{{")
        out.extend(f'\t"{name}",' for name in kind.features)
        out.append("}")
        out.append("")
        out.extend(_go_comment(f"{struct} is the {kind.kind_id} feature vector, in contract order. {kind.description}"))
        out.append(f"type {struct} struct {{")
        out.extend(f'\t{go_name(name)} float64 `json:"{name}"`' for name in kind.features)
        out.append("}")
        out.append("")
        out.append("// Map returns the features keyed by name, the shape RecommendRequest.Features carries.")
        out.append(f"func (f {struct}) Map() map[string]float64 {{")
        out.append("\treturn map[string]float64{")
        out.extend(f'\t\t"{name}": f.{go_name(name)},' for name in kind.features)
        out.append("\t}")
        out.append("}")
        out.append("")
        out.append("// Vector returns the features positionally, in contract order.")
        out.append(f"func (f {struct}) Vector() []float64 {{")
        out.append("\treturn []float64{" + ", ".join(f"f.{go_name(name)}" for name in kind.features) + "}")
        out.append("}")
        out.append("")

    for schema in spec.schemas:
        if schema.description:
            out.extend(_go_comment(f"{schema.name}: {schema.description}"))
        else:
            out.append(f"// {schema.name} is a wire schema of the engine.")
        out.append(f"type {schema.name} struct {{")
        for index, field in enumerate(schema.fields):
            lines = _go_field(spec, field)
            if index and len(lines) > 1:
                out.append("")
            out.extend(lines)
        out.append("}")
        out.append("")

    out.append("// Schemas constructs an empty value of every wire schema, by name.")
    out.append("var Schemas = map[string]func() any{")
    out.extend(f'\t"{s.name}": func() any {{ return new({s.name}) }},' for s in spec.schemas)
    out.extend(
        f'\t"{features_class(k.kind_id)}": func() any {{ return new({features_class(k.kind_id)}) }},'
        for k in spec.kinds
    )
    out.append("}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Formatting and output
# ---------------------------------------------------------------------------


def ruff_format(source: str) -> str:
    ruff = shutil.which("ruff", path=str(Path(sys.executable).parent)) or shutil.which("ruff")
    command = [ruff] if ruff else [sys.executable, "-m", "ruff"]
    result = subprocess.run(
        [*command, "format", "--stdin-filename", "generated.py", "-"],
        input=source,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise SpecError(f"ruff format failed: {result.stderr.strip()}")
    return result.stdout


def gofmt(source: str) -> str | None:
    """The gofmt-formatted source, or ``None`` when gofmt is not installed."""
    binary = shutil.which("gofmt")
    if binary is None:
        return None
    result = subprocess.run([binary], input=source, capture_output=True, text=True)
    if result.returncode != 0:
        raise SpecError(f"gofmt failed: {result.stderr.strip()}")
    return result.stdout


def generate(strict: bool) -> dict[Path, str]:
    spec = load_spec()
    outputs = {
        PY_VOCAB_PATH: ruff_format(render_python_vocab(spec)),
        PY_MODELS_PATH: ruff_format(render_python_models(spec)),
    }
    go_source = gofmt(render_go(spec))
    if go_source is not None:
        outputs[GO_PATH] = go_source
    elif strict:
        raise SpecError("gofmt is not on PATH; it is required with --strict")
    else:
        print(f"WARN: gofmt is not on PATH; {GO_PATH.relative_to(ROOT)} was not generated or checked.")
    return outputs


def main() -> None:
    check = "--check" in sys.argv
    try:
        outputs = generate(strict="--strict" in sys.argv)
    except SpecError as exc:
        print(f"FAIL: {exc}")
        sys.exit(1)

    if check:
        stale = [path for path, text in outputs.items() if not path.exists() or path.read_text("utf-8") != text]
        for path in stale:
            print(f"FAIL: {path.relative_to(ROOT)} is out of date. Run 'make typewriter' to update it.")
        if stale:
            sys.exit(1)
        print(f"OK: {len(outputs)} generated file(s) match typewriter/spec.yaml.")
        return

    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path.relative_to(ROOT)} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
