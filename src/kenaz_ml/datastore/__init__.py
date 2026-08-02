"""Data-access layer for kenaz-ml.

The `DataStore` protocol and its two implementations. Import from this
package, not from its submodules -- the submodule split is an implementation
detail that may change again (D-003, FR-010).

Implementations are deliberately *not* re-exported here: `SqliteStore` and
`PostgresStore` are selected through `create_store()`, and `PostgresStore`
pulls the optional `psycopg2` dependency that only cloud mode installs.
"""

from kenaz_ml.datastore.protocol import DataStore, create_store

__all__ = [
    "DataStore",
    "create_store",
]
