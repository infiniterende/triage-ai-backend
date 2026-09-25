"""
Resolve and normalise the Postgres connection URL for SQLAlchemy/psycopg2.

The same Supabase URL is often shared with Prisma (frontend), which appends
parameters psycopg2 does not understand (``pgbouncer=true``,
``connection_limit``, ``pool_timeout`` …). Render's own databases hand out a
``postgres://`` scheme that SQLAlchemy 2 no longer accepts. This helper makes
both work and gives a clear error when nothing is configured.
"""

from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters that belong to Prisma / other clients, not libpq.
_NON_LIBPQ_PARAMS = {
    "pgbouncer",
    "connection_limit",
    "pool_timeout",
    "schema",
    "connect_timeout_ms",
    "statement_cache_size",
}

_ENV_KEYS = ("SUPABASE_DATABASE_URL", "DATABASE_URL", "DIRECT_URL")


def normalise_database_url(url: str) -> str:
    url = url.strip().strip('"').strip("'")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    parts = urlsplit(url)
    params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k not in _NON_LIBPQ_PARAMS]
    # Supabase requires TLS; libpq defaults to "prefer", which is fine, but
    # being explicit avoids surprises behind some proxies.
    if parts.hostname and parts.hostname.endswith(("supabase.com", "supabase.co")) and not any(k == "sslmode" for k, _ in params):
        params.append(("sslmode", "require"))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(params), parts.fragment))


def get_database_url() -> str:
    for key in _ENV_KEYS:
        raw = os.getenv(key)
        if raw:
            return normalise_database_url(raw)
    raise RuntimeError(
        "No database URL configured. Set SUPABASE_DATABASE_URL (or DATABASE_URL) "
        "to your Postgres connection string, e.g. "
        "postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres"
    )
