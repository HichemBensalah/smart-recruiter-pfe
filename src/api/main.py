from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import candidates, chat, decision_cards, demo, graph, graph_neo4j, health, match
from src.core.matching.faiss_indexer import DEFAULT_SENTENCE_MODEL, _load_sentence_transformer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """API lifespan hook: preload SentenceTransformer model at startup."""
    # Startup
    try:
        _load_sentence_transformer(DEFAULT_SENTENCE_MODEL)
        print(f"[API] SentenceTransformer model '{DEFAULT_SENTENCE_MODEL}' preloaded at startup")
    except Exception as e:
        print(f"[API] Warning: SentenceTransformer preload failed: {e} (will lazy-load on first request)")

    yield

    # Shutdown (no cleanup needed for model)


app = FastAPI(
    title="Smart Recruiter API",
    description="API metier de demo pour exposer Matching V3, Decision Cards, ML comparison et Potential Graph.",
    version="demo",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(candidates.router)
app.include_router(match.router)
app.include_router(chat.router)
app.include_router(decision_cards.router)
app.include_router(graph.router)
app.include_router(graph_neo4j.router)
app.include_router(demo.router)
