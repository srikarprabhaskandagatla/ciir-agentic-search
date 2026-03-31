"""
Core data models for the Agentic Search pipeline.
Every value in the output table is wrapped in CellValue,
which carries its source URL, snippet, and confidence score.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime



# Source attribution — the core differentiator of this system
class SourceRef(BaseModel):
    """A traceable reference to the exact web content that produced a value."""
    url: str
    title: str = ""
    snippet: str = ""  # The exact excerpt from the page that contains this value


class CellValue(BaseModel):
    """A single table cell with full provenance."""
    value: Any
    sources: list[SourceRef] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @property
    def display_value(self) -> str:
        if self.value is None:
            return ""
        return str(self.value)



# Entity — a single row in the output table
class Entity(BaseModel):
    """One discovered entity (company, restaurant, tool, etc.)."""
    id: str
    cells: dict[str, CellValue]  # column_name to CellValue

    def get_name(self) -> str:
        cell = self.cells.get("name")
        return str(cell.value) if cell else self.id

    def coverage(self, columns: list[str]) -> float:
        """Fraction of columns that are filled."""
        filled = sum(1 for c in columns if c in self.cells and self.cells[c].value is not None)
        return filled / len(columns) if columns else 0.0



# Pipeline intermediate models
class SearchPlan(BaseModel):
    """Output of the Planner stage."""
    entity_type: str
    columns: list[str]        # Inferred schema columns (always starts with "name")
    search_queries: list[str] # Diverse queries to cover different angles
    rationale: str


class SearchResult(BaseModel):
    """A single result from the web search."""
    url: str
    title: str = ""
    snippet: str = ""


class ScrapedPage(BaseModel):
    """Cleaned content from a scraped URL."""
    url: str
    title: str = ""
    content: str
    error: Optional[str] = None



# Final output
class EntityTable(BaseModel):
    """The structured output table — the final product of the pipeline."""
    query: str
    entity_type: str
    columns: list[str]
    entities: list[Entity]
    sources_consulted: list[str]
    search_queries_used: list[str]
    rounds_completed: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def source_count(self) -> int:
        return len(self.sources_consulted)



# Streaming progress
class ProgressUpdate(BaseModel):
    """Sent over SSE to the frontend during pipeline execution."""
    stage: str       # planning | searching | scraping | extracting | resolving | analyzing | done
    message: str
    progress: float  # 0.0 to 1.0
    detail: Optional[str] = None