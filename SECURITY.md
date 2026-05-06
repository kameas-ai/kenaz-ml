# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| latest release | Yes |
| older releases | No |

kenaz-ml is pre-1.0. Only the latest release receives security fixes.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email **security@sigilos.io** with:

1. Description of the vulnerability
2. Steps to reproduce
3. Affected version(s)
4. Impact assessment (what an attacker could do)

You will receive an acknowledgment within 48 hours. We aim to provide a fix
or mitigation within 7 days for critical issues.

## Scope

kenaz-ml runs as a local Python process with no elevated privileges. The
security surface includes:

- **Local SQLite database** (`~/.local/share/sigild/data.db`) — contains
  workflow event data and ML predictions. Protected by filesystem permissions.
- **HTTP server** (`127.0.0.1:7774`) — binds to localhost only. Accepts
  prediction requests and training triggers. No authentication (local-only
  by design).
- **Model weights** (`~/.local/share/sigild/ml-models/`) — serialized
  scikit-learn models via joblib. Protected by filesystem permissions.

## Design Principles

- **Local-only.** kenaz-ml makes no outbound network calls. No telemetry,
  no external APIs, no cloud services.
- **Localhost binding.** The HTTP server binds to `127.0.0.1`, not `0.0.0.0`.
  It is not reachable from the network.
- **No secrets.** kenaz-ml has no API keys, tokens, or credentials.
- **No elevated privileges.** Runs as the current user. No root/sudo required.
- **Minimal dependencies.** Fewer dependencies means a smaller attack surface.
