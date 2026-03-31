"""
FastAPI Backend — Agentic Search API
Free stack: Groq (Llama 3.3 70B) + Tavily Search
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager

from groq import AsyncGroq

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.models import SearchResult, EntityTable
from backend.pipeline import scraper, resolver, gap_analyzer
from backend.pipeline.planner import plan_search
from backend.pipeline.searcher import search as tavily_search
from backend.pipeline.extractor import extract_from_pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv("backend/.env")

groq_client: AsyncGroq | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global groq_client
    groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", ""))
    logger.info("Client Started") # Groq Client
    yield
    await groq_client.close()
    logger.info("Client Stopped")


app = FastAPI(title="Agentic Search API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=300)
    max_rounds: int = Field(default=2, ge=1, le=3)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": app.version, "llm": "groq/llama-3.3-70b", "search": "tavily"}


@app.get("/api/example-queries")
async def example_queries():
    return {"examples": [
        "AI startups in healthcare",
        "top pizza places in Brooklyn",
        "open source database tools",
        "autonomous vehicle companies 2024",
        "no-code app building tools",
        "large language model providers",
    ]}


@app.post("/api/search")
async def search_endpoint(request: SearchRequest):
    if groq_client is None:
        raise HTTPException(status_code=503, detail="Groq client not initialised")

    queue: asyncio.Queue[dict | None] = asyncio.Queue()

    async def emit(stage, message, progress, detail=None):
        await queue.put({"type": "progress", "stage": stage, "message": message,
                         "progress": min(progress, 1.0), "detail": detail})

    async def run():
        try:
            all_entities = []
            seen_urls: set[str] = set()
            all_search_results: list[SearchResult] = []
            all_queries_used: list[str] = []

            # Plan
            await emit("planning", "Analysing query and inferring schema...", 0.04)
            plan = await plan_search(groq_client, request.query)
            await emit("planning",
                       f"Schema: {len(plan.columns)} columns for '{plan.entity_type}'",
                       0.10, f"Columns: {', '.join(plan.columns)}")

            queries_this_round = plan.search_queries
            completed_rounds = 0

            for round_idx in range(request.max_rounds):
                completed_rounds = round_idx + 1
                rl = f"Round {round_idx+1}/{request.max_rounds}"
                rs = 0.10 + round_idx * (0.85 / request.max_rounds)
                rspan = 0.85 / request.max_rounds
                rp = lambda f, _rs=rs, _rspan=rspan: _rs + f * _rspan

                # Search
                await emit("searching", f"{rl}: Searching with {len(queries_this_round)} queries...", rp(0.05))
                search_tasks = [tavily_search(q, max_results=6) for q in queries_this_round]
                results_per_query = await asyncio.gather(*search_tasks, return_exceptions=True)
                all_queries_used.extend(queries_this_round)

                new_results: list[SearchResult] = []
                for results in results_per_query:
                    if isinstance(results, Exception):
                        logger.warning("Search task failed: %s", results)
                        continue
                    for r in results:
                        if r.url not in seen_urls:
                            seen_urls.add(r.url)
                            new_results.append(r)
                            all_search_results.append(r)

                await emit("searching", f"{rl}: Found {len(new_results)} new URLs", rp(0.18))
                if not new_results:
                    break

                # Scrape
                await emit("scraping", f"{rl}: Fetching {len(new_results)} pages...", rp(0.22))
                pages = await scraper.scrape_urls([r.url for r in new_results])
                await emit("scraping", f"{rl}: Scraped {len(pages)} pages successfully", rp(0.40))
                if not pages:
                    break

                # Extract
                await emit("extracting", f"{rl}: Extracting entities with Llama 3.3...", rp(0.45))
                new_raw = await extract_from_pages(groq_client, pages, plan.columns, plan.entity_type)
                await emit("extracting", f"{rl}: Found {len(new_raw)} entity mentions", rp(0.65))
                all_entities.extend(new_raw)

                # Resolve
                await emit("resolving", f"{rl}: Deduplicating and merging...", rp(0.70))
                all_entities = await resolver.resolve_entities(groq_client, all_entities)
                await emit("resolving", f"{rl}: {len(all_entities)} unique entities", rp(0.80))

                # Gap Analysis
                if round_idx < request.max_rounds - 1:
                    await emit("analyzing", "Checking data coverage...", rp(0.88))
                    gap = await gap_analyzer.analyze_gaps(
                        groq_client, all_entities, plan.columns,
                        request.query, plan.entity_type
                    )
                    if gap.get("should_continue") and gap.get("gap_queries"):
                        queries_this_round = gap["gap_queries"]
                        await emit("analyzing",
                                   f"Gaps found - running follow-up: {gap.get('gap_summary', '')}",
                                   rp(0.92))
                    else:
                        await emit("analyzing", "Coverage sufficient - stopping early", rp(0.92))
                        break

            await emit("done",
                       f"Done! {len(all_entities)} entities - "
                       f"{len(all_search_results)} sources - "
                       f"{completed_rounds} round(s)",
                       1.0)

            result = EntityTable(
                query=request.query,
                entity_type=plan.entity_type,
                columns=plan.columns,
                entities=all_entities,
                sources_consulted=[r.url for r in all_search_results],
                search_queries_used=all_queries_used,
                rounds_completed=completed_rounds,
            )
            await queue.put({"type": "result", "data": result.model_dump(mode="json")})

        except Exception as exc:
            logger.exception("Pipeline error for query: %s", request.query)
            await queue.put({"type": "error", "message": str(exc)})
        finally:
            await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(run())
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=4.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0",
                port=int(os.getenv("PORT", 8000)), reload=True)