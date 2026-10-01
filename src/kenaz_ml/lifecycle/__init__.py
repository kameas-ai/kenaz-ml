"""Engine lifecycle: client leases, self-termination, token-authorized shutdown.

two-client-engine-01MSK2EN WP05. Standard library only (NFR-001); no outbound
network, no signature verification (C-001, C-003), and nothing here ever
creates or writes into the client-owned install root (C-004).
"""
