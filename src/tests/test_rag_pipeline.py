from __future__ import annotations

from src.rag.answerer import ExtractiveAnswerer
from src.rag.embeddings import HashEmbedder
from src.rag.pipeline import RAGPipeline
from src.rag.types import Document, SearchResult
from src.vectorstore.inmemory import InMemoryVectorStore


def build_pipeline(min_score: float = 0.2) -> RAGPipeline:
    embedder = HashEmbedder()
    vectorstore = InMemoryVectorStore(embedder=embedder)
    answerer = ExtractiveAnswerer(max_chars=200)
    return RAGPipeline(vectorstore=vectorstore, answerer=answerer, min_score=min_score)


def test_retrieval_returns_grounded_answer() -> None:
    pipeline = build_pipeline()
    documents = [
        Document(doc_id="sales", content="Q4 sales were 100 units in North India.", metadata={"source": "report"}),
        Document(doc_id="hr", content="Employees joined after June 2023: Alice, Bob.", metadata={"source": "hr"}),
    ]
    pipeline.ingest(documents)

    response = pipeline.answer("What were Q4 sales in North India?")

    assert "Q4 sales were 100 units" in response.answer
    assert response.sources
    assert response.refusal_reason is None


def test_empty_context_refuses() -> None:
    pipeline = build_pipeline(min_score=0.99)
    response = pipeline.answer("What is the policy on travel?")

    assert response.answer.startswith("I don't know")
    assert response.sources == []
    assert response.refusal_reason in {"no_context", "empty_context"}


def test_min_score_filters_low_relevance() -> None:
    pipeline = build_pipeline(min_score=0.9)
    documents = [
        Document(doc_id="general", content="General policy document.", metadata={"source": "policy"}),
        Document(doc_id="travel", content="Travel policy covers flights and hotels.", metadata={"source": "policy"}),
    ]
    pipeline.ingest(documents)

    response = pipeline.answer("What is the travel policy?", min_score=0.99)

    assert response.answer.startswith("I don't know")
    assert response.sources == []


def test_source_name_weight_reranks() -> None:
    pipeline = build_pipeline()
    pipeline.source_name_weights = {"beta": 2.0}
    results = [
        SearchResult(
            document=Document(
                doc_id="alpha-doc",
                content="Alpha content",
                metadata={"source_type": "db", "source_name": "alpha"},
            ),
            score=0.4,
        ),
        SearchResult(
            document=Document(
                doc_id="beta-doc",
                content="Beta content",
                metadata={"source_type": "db", "source_name": "beta"},
            ),
            score=0.4,
        ),
    ]

    contexts = pipeline.build_context(results, limit=1)

    assert contexts
    assert contexts[0].document_id == "beta-doc"


def test_min_score_by_source_overrides_type() -> None:
    pipeline = build_pipeline()
    results = [
        SearchResult(
            document=Document(
                doc_id="finance-doc",
                content="Finance content",
                metadata={"source_type": "db", "source_name": "finance"},
            ),
            score=0.5,
        ),
        SearchResult(
            document=Document(
                doc_id="sales-doc",
                content="Sales content",
                metadata={"source_type": "db", "source_name": "sales"},
            ),
            score=0.5,
        ),
    ]

    contexts = pipeline.build_context(
        results,
        min_score_by_type={"db": 0.1},
        min_score_by_source={"finance": 0.9},
        limit=2,
    )

    assert [chunk.document_id for chunk in contexts] == ["sales-doc"]
