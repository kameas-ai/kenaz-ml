package typewriter

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// stripNulls removes null-valued object members, so an omitted optional
// field and an explicit null compare equal.
func stripNulls(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, e := range t {
			if e != nil {
				out[k] = stripNulls(e)
			}
		}
		return out
	case []any:
		for i, e := range t {
			t[i] = stripNulls(e)
		}
	}
	return v
}

// TestFixturesRoundTrip decodes every testdata/<Schema>.<case>.json (captured
// from a running engine) into its generated type, refusing unknown fields, and
// checks that re-encoding loses nothing.
func TestFixturesRoundTrip(t *testing.T) {
	paths, err := filepath.Glob("testdata/*.json")
	if err != nil || len(paths) == 0 {
		t.Fatalf("no fixtures found: %v", err)
	}
	for _, path := range paths {
		name := strings.SplitN(filepath.Base(path), ".", 2)[0]
		t.Run(filepath.Base(path), func(t *testing.T) {
			construct, ok := Schemas[name]
			if !ok {
				t.Fatalf("fixture names unknown schema %q", name)
			}
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			value := construct()
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.DisallowUnknownFields()
			if err := dec.Decode(value); err != nil {
				t.Fatalf("decode: %v", err)
			}
			encoded, err := json.Marshal(value)
			if err != nil {
				t.Fatal(err)
			}
			var want, got any
			if err := json.Unmarshal(raw, &want); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(encoded, &got); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(stripNulls(want), stripNulls(got)) {
				t.Errorf("round trip changed the payload:\nwant %s\ngot  %s", raw, encoded)
			}
		})
	}
}

// TestFeatureStructsMatchTheContract pins each feature struct to its kind's
// ordered names: Map has exactly the names, Vector follows their order.
func TestFeatureStructsMatchTheContract(t *testing.T) {
	cases := map[string]interface {
		Map() map[string]float64
		Vector() []float64
	}{
		KindBranchNow:     BranchNowFeatures{},
		KindCompactNow:    CompactNowFeatures{},
		KindEscalateModel: EscalateModelFeatures{},
	}
	if len(cases) != len(KindIDs) {
		t.Fatalf("%d kinds are published but %d are covered here", len(KindIDs), len(cases))
	}
	for kind, zero := range cases {
		names := FeatureNames[kind]
		// Give field i the value i+1 through JSON, so position is observable.
		payload := map[string]float64{}
		for i, name := range names {
			payload[name] = float64(i + 1)
		}
		raw, _ := json.Marshal(payload)
		filled := reflect.New(reflect.TypeOf(zero))
		dec := json.NewDecoder(bytes.NewReader(raw))
		dec.DisallowUnknownFields()
		if err := dec.Decode(filled.Interface()); err != nil {
			t.Fatalf("%s: %v", kind, err)
		}
		features := filled.Elem().Interface().(interface {
			Map() map[string]float64
			Vector() []float64
		})
		if !reflect.DeepEqual(features.Map(), payload) {
			t.Errorf("%s: Map() = %v, want %v", kind, features.Map(), payload)
		}
		for i, v := range features.Vector() {
			if v != float64(i+1) {
				t.Errorf("%s: Vector()[%d] = %v, want %d (%s)", kind, i, v, i+1, names[i])
			}
		}
		if len(ContractVersions[kind]) != 16 {
			t.Errorf("%s: contract version %q is not 16 hex characters", kind, ContractVersions[kind])
		}
	}
}
