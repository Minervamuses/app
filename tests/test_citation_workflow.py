import asyncio
from pathlib import Path

from citation import bibtex
from citation import capture
from citation.discovery import parse_summaries
from citation.models import PaperCandidate


def test_parse_summaries_cleans_search_labels_and_extracts_doi():
    text = """Search summaries:

**1. arXiv arxiv.org › abs  › 1706.03762   [1706.03762] Attention Is All You Need**
URL: https://arxiv.org/abs/1706.03762
Description: DOI: 10.48550/arXiv.1706.03762

---

**2. ACM Digital Library dl.acm.org › doi  › 10.5555  › 3295222.3295349   Attention is all you need | Proceedings of the 31st International Conference on Neural Information Processing Systems**
URL: https://dl.acm.org/doi/10.5555/3295222.3295349
Description: No description available

---

**3. Google Scholar**
URL: https://scholar.google.com/scholar_lookup?title=Attention+is+all+you+need&amp=&author=A.+Vaswani&amp=&publication_year=2017&amp=&doi=10.48550/arXiv.1706.03762
Description: No description available

---

**4. Ashish Vaswani**
URL: https://scholar.google.com/citations?user=oR9sCGYAAAAJ&hl=en
Description: No description available

---

**5. Attention Is All You Need - Wikipedia**
URL: https://en.wikipedia.org/wiki/Attention_Is_All_You_Need
Description: No description available
"""

    candidates = parse_summaries(text)

    assert candidates[0].title == "Attention Is All You Need"
    assert candidates[0].doi == "10.48550/arXiv.1706.03762"
    assert candidates[1].title == "Attention is all you need"
    assert candidates[1].doi == "10.5555/3295222.3295349"
    assert candidates[2].title == "Attention is all you need"
    assert candidates[2].doi == "10.48550/arXiv.1706.03762"
    assert candidates[2].year == 2017
    assert candidates[2].authors == ["A. Vaswani"]
    assert len(candidates) == 3


def test_capture_retries_alternate_doi_through_crossref(monkeypatch, tmp_path):
    class Runtime:
        @property
        def cite_dir(self) -> Path:
            return tmp_path

    resolve_calls = []

    async def fake_resolve_doi(
        runtime,
        candidate,
        *,
        confirm_cb,
            result,
            allow_direct=True,
            exclude=None,
            progress_cb=None,
        ):
        resolve_calls.append((allow_direct, set(exclude or set())))
        if allow_direct:
            return "10.bad/first"
        return "10.good/second"

    def fake_fetch_bibtex_for_doi(doi):
        if doi == "10.bad/first":
            return None, ["bad DOI failed"]
        return (
            "@article{good,\n"
            "  title = {Good Paper},\n"
            "}\n",
            ["good DOI retrieved"],
        )

    monkeypatch.setattr(capture, "_resolve_doi", fake_resolve_doi)
    monkeypatch.setattr(capture, "fetch_bibtex_for_doi", fake_fetch_bibtex_for_doi)

    result = asyncio.run(
        capture.capture_citation(Runtime(), PaperCandidate("Good Paper"))
    )

    assert result.ok is True
    assert result.route == "crossref"
    assert result.doi == "10.good/second"
    assert Path(result.out_path).read_text(encoding="utf-8").startswith("@article")
    assert resolve_calls == [
        (True, set()),
        (False, {"10.bad/first"}),
    ]
    assert "bad DOI failed" in result.notes
    assert "good DOI retrieved" in result.notes


def test_bibtex_title_extraction_handles_single_line_crossref_bibtex():
    one_line = (
        "@article{Mohaidat_2024, title={A Survey on Neural Network Hardware "
        "Accelerators}, volume={5}, ISSN={2691-4581}, "
        "url={http://dx.doi.org/10.1109/TAI.2024.3377147}, "
        "DOI={10.1109/tai.2024.3377147}, number={8}, journal={IEEE "
        "Transactions on Artificial Intelligence}, publisher={Institute of "
        "Electrical and Electronics Engineers (IEEE)}, author={Mohaidat, "
        "Tamador and Khalil, Kasem}, year={2024}, month=aug, pages={3801-3822} }"
    )

    assert (
        bibtex.extract_bibtex_title(one_line)
        == "A Survey on Neural Network Hardware Accelerators"
    )
    assert (
        bibtex.normalize_title_to_filename(bibtex.extract_bibtex_title(one_line))
        == "a_survey_on_neural_network_hardware_accelerators.bib"
    )


def test_normalized_bibtex_filename_is_length_capped():
    filename = bibtex.normalize_title_to_filename("x" * 400)

    assert filename.endswith(".bib")
    assert len(filename.removesuffix(".bib")) <= 160
