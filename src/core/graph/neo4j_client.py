from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable


class Neo4jUnavailable(RuntimeError):
    """Raised when Neo4j cannot be used in the current environment."""


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str


def load_neo4j_settings() -> Neo4jSettings:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    missing = [name for name, value in {"NEO4J_URI": uri, "NEO4J_USER": user, "NEO4J_PASSWORD": password}.items() if not value]
    if missing:
        raise Neo4jUnavailable(f"Neo4j is not configured. Missing environment variables: {', '.join(missing)}")
    return Neo4jSettings(uri=str(uri), user=str(user), password=str(password))


class Neo4jClient:
    def __init__(self, settings: Neo4jSettings | None = None) -> None:
        self.settings = settings or load_neo4j_settings()
        self._driver: Any | None = None

    def __enter__(self) -> "Neo4jClient":
        self.connect()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def connect(self) -> None:
        if self._driver is not None:
            return
        try:
            from neo4j import GraphDatabase
        except ImportError as exc:
            raise Neo4jUnavailable("Neo4j Python driver is not installed. Install dependency: neo4j") from exc
        try:
            self._driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, self.settings.password),
            )
            self._driver.verify_connectivity()
        except Exception as exc:
            self._driver = None
            raise Neo4jUnavailable(f"Neo4j is unavailable: {exc}") from exc

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def execute_read(self, callback: Callable[..., Any], **kwargs: Any) -> Any:
        self.connect()
        with self._driver.session() as session:
            return session.execute_read(callback, **kwargs)

    def execute_write(self, callback: Callable[..., Any], **kwargs: Any) -> Any:
        self.connect()
        with self._driver.session() as session:
            return session.execute_write(callback, **kwargs)


def neo4j_status() -> dict[str, Any]:
    try:
        with Neo4jClient():
            return {"neo4j_available": True, "message": "Neo4j is configured and reachable."}
    except Neo4jUnavailable as exc:
        return {"neo4j_available": False, "message": str(exc)}
