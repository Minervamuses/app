"""End-to-end evaluator — tests retrieval, agent behavior, and answer quality.

Generates multi-chunk synthesis questions with reference answers, runs them
through the full agent graph, then uses LLM-as-judge to score the output.
"""

import json
import random
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from kms.agent.graph import build_graph
from kms.cli.chat import SYSTEM_PROMPT
from kms.config import KMSConfig
from kms.evaluation.base import BaseEvaluator, EvalResult, _extract_json
from kms.llm.openrouter import OpenRouterLLM
from kms.store.json_store import JSONStore

GENERATE_PROMPT = """You are a test-case generator. Below are several related chunks from a knowledge base. Generate a question that requires synthesizing information from multiple chunks to answer.

{chunk_texts}

Return a JSON object with:
1. "question": a question that needs information from at least 2 of these chunks
2. "reference_answer": a 3-5 sentence answer based only on the provided chunks
3. "required_chunks": list of objects with "pid" and "chunk_id" that are essential to answer
4. "question_type": one of "explore_first", "direct_search", "multi_search", "needs_context"

Return ONLY the JSON object, no explanation."""

JUDGE_PROMPT = """You are an answer quality judge. Compare an agent's answer against a reference answer.

Question: {question}
Reference answer: {reference_answer}
Agent's answer: {actual_answer}

Score the agent's answer (0-3):
0 = completely wrong or hallucinated
1 = partially correct but missing key information
2 = mostly correct, covers the main points
3 = complete and accurate

Return a JSON object with:
1. "score": integer 0-3
2. "rationale": brief explanation

Return ONLY the JSON object."""


class EndToEndEvaluator(BaseEvaluator):
    """Evaluate the full pipeline: retrieval + agent behavior + answer quality."""

    def __init__(self, config: KMSConfig | None = None):
        self.config = config or KMSConfig()
        self._llm = OpenRouterLLM(config=self.config)

    def generate(self, n: int = 15, output_path: str | None = None) -> list[dict]:
        """Generate synthesis questions from groups of related chunks.

        Samples chunks that share the same folder (likely related), then asks
        the LLM to create a question requiring information from multiple chunks.

        Args:
            n: Number of test cases to generate.
            output_path: Optional path to save generated cases as JSON.

        Returns:
            List of test case dicts with question, reference_answer,
            required_chunks, and question_type.
        """
        json_store = JSONStore(self.config.raw_json_path())
        all_docs = json_store.get()

        # Group chunks by folder for relatedness
        by_folder: dict[str, list] = {}
        for doc in all_docs:
            folder = doc.metadata.get("folder", "")
            by_folder.setdefault(folder, []).append(doc)

        # Only folders with 2+ chunks can produce synthesis questions
        eligible = {f: docs for f, docs in by_folder.items() if len(docs) >= 2}
        if not eligible:
            return []

        folders = list(eligible.keys())
        cases = []

        for _ in range(n):
            if not folders:
                break
            folder = random.choice(folders)
            docs = eligible[folder]
            sample_size = min(random.randint(2, 4), len(docs))
            sampled = random.sample(docs, sample_size)

            chunk_texts = ""
            for i, doc in enumerate(sampled, 1):
                meta = doc.metadata
                chunk_texts += (
                    f"\n[Chunk {i} - pid: {meta.get('pid')}, "
                    f"chunk_id: {meta.get('chunk_id')}, "
                    f"file: {meta.get('file_path', '?')}]\n"
                    f"{doc.page_content[:1200]}\n"
                )

            prompt = GENERATE_PROMPT.format(chunk_texts=chunk_texts)

            try:
                response = self._llm.invoke(prompt, max_tokens=500, temperature=0.0)
                data = _extract_json(response)
            except (ValueError, json.JSONDecodeError):
                continue

            cases.append({
                "question": data.get("question", ""),
                "reference_answer": data.get("reference_answer", ""),
                "required_chunks": data.get("required_chunks", []),
                "question_type": data.get("question_type", "direct_search"),
                "source_folder": folder,
            })

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cases, f, ensure_ascii=False, indent=2)

        return cases

    def evaluate(self, cases: list[dict]) -> EvalResult:
        """Run each case through the agent graph and judge answer quality.

        Args:
            cases: Test cases from generate() or loaded from JSON.

        Returns:
            EvalResult with avg_score (0-3), score_distribution, and per-case details.
        """
        graph = build_graph(self.config)

        total_score = 0
        score_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        details = []

        for case in cases:
            thread_id = str(uuid.uuid4())
            run_config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 16,
            }

            result = graph.invoke(
                {"messages": [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=case["question"]),
                ]},
                config=run_config,
            )

            actual_answer = result["messages"][-1].content or ""

            # LLM-as-judge
            judge_prompt = JUDGE_PROMPT.format(
                question=case["question"],
                reference_answer=case["reference_answer"],
                actual_answer=actual_answer,
            )

            try:
                judge_response = self._llm.invoke(judge_prompt, max_tokens=200, temperature=0.0)
                judge_data = _extract_json(judge_response)
                score = int(judge_data.get("score", 0))
                score = max(0, min(3, score))
                rationale = judge_data.get("rationale", "")
            except (ValueError, json.JSONDecodeError):
                score = 0
                rationale = "Judge parsing failed"

            total_score += score
            score_dist[score] += 1
            details.append({
                "question": case["question"],
                "question_type": case.get("question_type"),
                "reference_answer": case["reference_answer"],
                "actual_answer": actual_answer[:500],
                "score": score,
                "rationale": rationale,
            })

        total = len(cases)
        avg = total_score / total if total else 0

        return EvalResult(
            name="End-to-End",
            total=total,
            scores={
                "avg_score": avg / 3,  # Normalize to 0-1 for consistency
                "avg_score_raw": avg,  # Raw 0-3 scale
                "score_0_pct": score_dist[0] / total if total else 0,
                "score_1_pct": score_dist[1] / total if total else 0,
                "score_2_pct": score_dist[2] / total if total else 0,
                "score_3_pct": score_dist[3] / total if total else 0,
            },
            details=details,
        )
