from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import candidates, chat, decision_cards, demo, graph, graph_neo4j, health, match


app = FastAPI(
    title="Smart Recruiter API",
    description="API metier de demo pour exposer Matching V3, Decision Cards, ML comparison et Potential Graph.",
    version="demo",
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
