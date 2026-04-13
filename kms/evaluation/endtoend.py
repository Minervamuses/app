"""End-to-end evaluator — tests retrieval, agent behavior, and answer quality.

Generates multi-chunk synthesis questions with reference answers, runs them
through the full agent graph, then uses LLM-as-judge to score the output.
"""

import json
import random
from pathlib import Path

from langgraph.errors import GraphRecursionError

from kms.cli.chat import ChatSession
from kms.config import KMSConfig
from kms.evaluation.base import BaseEvaluator, EvalResult, _extract_json
from kms.llm.ollama import OllamaLLM
from kms.llm.openrouter import OpenRouterLLM
from kms.store.json_store import JSONStore

FILTER_PROMPT = """Is this text chunk semantically meaningful content that a human would ask questions about? Answer YES or NO only.

Reject: compiled/minified code, lock files, binary data, auto-generated IDs, SQL DDL boilerplate, empty/trivial files.
Accept: documentation, source code with logic, research notes, config with explanations, test cases with assertions.

{chunk_text}"""

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
        self._filter_llm = OllamaLLM(model_name=self.config.filter_llm_model, config=self.config)
        self._gen_llm = OpenRouterLLM(model_name=self.config.gen_llm_model, config=self.config)
        self._judge_llm = OpenRouterLLM(model_name=self.config.judge_llm_model, config=self.config)

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

            # Filter out non-semantic chunks via local LLM
            filtered = []
            for doc in sampled:
                preview = doc.page_content[:400]
                verdict = self._filter_llm.invoke(
                    FILTER_PROMPT.format(chunk_text=preview),
                    max_tokens=8,
                    temperature=0.0,
                )
                if verdict.strip().upper().startswith("YES"):
                    filtered.append(doc)
            if len(filtered) < 2:
                continue
            sampled = filtered

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
                response = self._gen_llm.invoke(prompt, max_tokens=4096, temperature=0.0)
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
        total_score = 0
        score_dist = {0: 0, 1: 0, 2: 0, 3: 0}
        details = []

        for case in cases:
            session = ChatSession(self.config, recursion_limit=32)

            try:
                actual_answer = session.turn(case["question"])
            except GraphRecursionError:
                actual_answer = "(agent hit recursion limit)"
            except Exception as exc:
                actual_answer = f"(agent error: {type(exc).__name__}: {exc})"

            # LLM-as-judge
            judge_prompt = JUDGE_PROMPT.format(
                question=case["question"],
                reference_answer=case["reference_answer"],
                actual_answer=actual_answer,
            )

            try:
                judge_response = self._judge_llm.invoke(judge_prompt, max_tokens=200, temperature=0.0)
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

        scores: dict[str, float] = {
            "avg_score": avg / 3,  # Normalize to 0-1 for consistency
            "avg_score_raw": avg,  # Raw 0-3 scale
        }
        scores.update({
            "score_0_pct": score_dist[0] / total if total else 0,
            "score_1_pct": score_dist[1] / total if total else 0,
            "score_2_pct": score_dist[2] / total if total else 0,
            "score_3_pct": score_dist[3] / total if total else 0,
        })

        return EvalResult(
            name="End-to-End",
            total=total,
            scores=scores,
            details=details,
        )
