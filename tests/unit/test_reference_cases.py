import json

from validation.reference import cases


def test_reference_corpus_is_unique_and_covers_types():
    corpus = list(cases())
    assert len(corpus) >= 100
    assert len({json.dumps(case["request"], sort_keys=True) for case in corpus}) == len(corpus)
    assert {
        question["type"] for case in corpus for question in case["request"]["questions"].values()
    } == {"choice", "noul", "score"}
    assert any("我" in str(case["request"]["state"]) for case in corpus)
