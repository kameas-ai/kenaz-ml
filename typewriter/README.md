# typewriter

The engine's wire contract, written once in [`spec.yaml`](spec.yaml) and
generated for each language that speaks it.

| Output | For | Generated file |
|---|---|---|
| Python vocabularies (standard library only) | the engine | `src/kenaz_ml/typewriter/vocab.py` |
| Python pydantic models | the engine's routes | `src/kenaz_ml/typewriter/models.py` |
| Go module `github.com/kameas-ai/kenaz-ml/typewriter` | Go clients | `typewriter.gen.go` |

Never edit a generated file. Change `spec.yaml`, then:

```bash
make typewriter        # regenerate (needs gofmt for the Go file)
make typewriter-check  # what CI runs: fail if anything is out of date
make typewriter-test   # Go conformance tests over testdata/
make openapi-check     # the HTTP description must still match
```

## What the spec owns

- **Schemas**: every request and response body of the engine's HTTP surface.
- **Kinds**: each recommendation kind's *ordered* feature names. The order is
  the vector layout, and the kind's contract version is a hash of it.
- **Enums**: the closed vocabularies (error kinds, user actions, backends,
  refusal reasons).
- **`vocabulary_version`**: the semantic salt. Types catch a renamed, added,
  removed or reordered feature. They cannot catch a feature whose *meaning*
  changed while its name stayed. That is what this number is for: bump it and
  every contract version moves.

## Using the Go module

```go
import "github.com/kameas-ai/kenaz-ml/typewriter"

features := typewriter.BranchNowFeatures{TurnsSinceSessionStart: 12 /* ... */}
req := typewriter.RecommendRequest{
    Features:               features.Map(),
    FeatureContractVersion: typewriter.BranchNowContractVersion,
}
```

A feature struct that does not match the engine no longer compiles, and
`BranchNowContractVersion` is the version that struct was generated against.
The engine still checks the version at runtime (`GET /v1/contracts`), which is
what protects a client built against an older spec.

Releases are tagged `typewriter/vX.Y.Z` (the Go convention for a module in a
subdirectory).

## Fixtures

`testdata/<Schema>.<case>.json` are payloads captured from a running engine.
The Go tests decode each one into its generated type with unknown fields
refused; `tests/test_typewriter.py` validates the same files against the
pydantic models and checks that a live engine still produces those shapes.

## Spec dialect

```yaml
schemas:
  Name:
    description: |          # becomes the docstring / Go doc comment
      ...
    exactly_one_of: [a, b]   # optional: exactly one of these fields is non-null
    fields:
      plain: string          # short form
      full:
        type: "map<list<string>>"
        nullable: true       # the value may be null
        default: null        # present = optional; absent = required
        description: ...     # published in the OpenAPI document
        comment: ...         # code comment only
        vocabulary: UserAction  # an open string whose known values are an enum
        min_length: 1        # also gt, ge, le
```

Types: `string`, `int`, `float`, `bool`, `any`, `object` (free-form JSON
object), `list<T>`, `map<T>` (string keys), a schema or enum name, and unions
written `A | B` (a union is `json.RawMessage` in Go). A field typed with an
enum name is validated against it; a field with `vocabulary:` stays an open
string, for values the engine answers with a typed refusal rather than a
validation error.
