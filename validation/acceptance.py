"""CPU-only answer comparison. A policy pass is not production certification.

Ties always select the first candidate in declared order. Noul uses ordered
[false, true] probabilities [1-p, p], hence true wins only when p > 0.5.
"""

import math
from decimal import Decimal

_NORMALIZATION_TOLERANCE = Decimal("0.000001")
_ROUNDING_SLACK = Decimal("0.000000000001")


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + " must be a finite JSON number")
    return Decimal(str(value))


def _probability(value):
    number = _number(value, "probability")
    if not 0 <= number <= 1:
        raise ValueError("probability outside [0,1]")
    return number


def _rounding(count, decimals):
    half_unit = Decimal(0) if decimals is None else Decimal(5).scaleb(-decimals - 1)
    return (
        half_unit,
        _NORMALIZATION_TOLERANCE + count * half_unit,
        _NORMALIZATION_TOLERANCE + (count * (count - 1) // 2 + 1) * half_unit,
    )


def _answer(answer, decimals=None):
    if not isinstance(answer, dict):
        raise ValueError("answer must be an object")
    if "confidence" in answer:
        _probability(answer["confidence"])
    fields = set(answer) - {"confidence"}
    kind = answer.get("type")
    if kind == "noul":
        if fields not in ({"type", "noul"}, {"type", "noul", "probabilities"}):
            raise ValueError("Noul requires only type and noul")
        p = _probability(answer["noul"])
        if "probabilities" in answer:
            mapping = answer["probabilities"]
            if not isinstance(mapping, dict) or list(mapping) != ["false", "true"]:
                raise ValueError("Noul distribution requires ordered false,true")
            values = [_probability(v) for v in mapping.values()]
            if abs(sum(values) - 1) > _NORMALIZATION_TOLERANCE or any(
                abs(a - b) > _NORMALIZATION_TOLERANCE
                for a, b in zip(values, [1 - p, p], strict=True)
            ):
                raise ValueError("Noul distribution contradicts scalar")
        return kind, ["false", "true"], [1 - p, p], None
    if kind not in ("choice", "score"):
        raise ValueError("unknown answer type")
    mapping = answer.get("probabilities")
    if (
        not isinstance(mapping, dict)
        or not mapping
        or any(not isinstance(k, str) or not k for k in mapping)
    ):
        raise ValueError("probabilities require nonempty ordered string candidates")
    keys, values = list(mapping), [_probability(p) for p in mapping.values()]
    _, mass_tolerance, score_tolerance = _rounding(len(values), decimals)
    if decimals is not None and any(v != v.quantize(Decimal(1).scaleb(-decimals)) for v in values):
        raise ValueError("reference probabilities exceed declared decimal places")
    if abs(sum(values) - 1) > mass_tolerance:
        raise ValueError("probabilities must sum to one; no normalization applied")
    maximum = max(range(len(values)), key=values.__getitem__)
    if kind == "choice":
        if fields != {"type", "probabilities", "choice"} or answer.get("choice") != keys[maximum]:
            raise ValueError("choice must match first argmax")
        return kind, keys, values, None
    if fields != {"type", "probabilities", "score", "legend"}:
        raise ValueError("Score fields differ from contract")
    legend = answer["legend"]
    if (
        len(keys) < 2
        or keys != [str(i) for i in range(len(keys))]
        or not isinstance(legend, dict)
        or list(legend) != keys
        or any(not isinstance(v, str) or not v for v in legend.values())
    ):
        raise ValueError("Score requires ordered ordinal candidates and matching legend")
    score = _number(answer["score"], "score")
    expected_score = sum(i * p for i, p in enumerate(values))
    if score < 0 or abs(score - expected_score) > score_tolerance:
        raise ValueError("score must equal its distribution expected value")
    return kind, keys, values, expected_score


def compare_answers(
    actual: dict, expected: dict, policy: dict, *, reference_probability_decimals=None
) -> dict:
    """Validate both answers, then compare against expected probabilities.

    Each probability must satisfy abs(actual-expected) <= atol + rtol*abs(expected),
    with fixed 1e-12 rounding slack. Score errors and total variation are metrics,
    not additional gates. Explicit reference decimal rounding widens only reference
    normalization and Score consistency validation, never probability gates or actual
    validation. expected_score_error uses reconstructed distribution expectations;
    reported_score_error compares returned Score.score fields. No renormalization.
    Invalid input raises ValueError instead of a policy fail.
    """
    if reference_probability_decimals is not None and (
        type(reference_probability_decimals) is not int
        or not 0 <= reference_probability_decimals <= 8
    ):
        raise ValueError("reference_probability_decimals must be integer 0..8 or None")
    fields = {"policy_id", "probability_atol", "probability_rtol", "require_argmax_match"}
    if not isinstance(policy, dict) or set(policy) != fields:
        raise ValueError("policy fields differ")
    if (
        not isinstance(policy["policy_id"], str)
        or not policy["policy_id"].strip()
        or type(policy["require_argmax_match"]) is not bool
    ):
        raise ValueError("invalid policy identifier or argmax requirement")
    atol = _number(policy["probability_atol"], "probability_atol")
    rtol = _number(policy["probability_rtol"], "probability_rtol")
    if atol < 0 or rtol < 0:
        raise ValueError("negative tolerance")
    if (
        not isinstance(actual, dict)
        or not isinstance(expected, dict)
        or not expected
        or set(actual) != set(expected)
        or any(not isinstance(q, str) or not q for q in expected)
    ):
        raise ValueError("answer question IDs must be equal nonempty string sets")
    questions = []
    for qid in expected:
        ak, keys, av, ascore = _answer(actual[qid])
        ek, ekeys, ev, escore = _answer(expected[qid], reference_probability_decimals)
        if ak != ek or keys != ekeys:
            raise ValueError("question type or ordered candidates differ")
        if ak == "score" and actual[qid]["legend"] != expected[qid]["legend"]:
            raise ValueError("Score legends differ")
        ai = max(range(len(av)), key=av.__getitem__)
        ei = max(range(len(ev)), key=ev.__getitem__)
        errors = [abs(a - e) for a, e in zip(av, ev, strict=True)]
        probability_passed = all(
            error <= atol + rtol * abs(e) + _ROUNDING_SLACK
            for error, e in zip(errors, ev, strict=True)
        )
        changed = ai != ei
        half_unit, mass_tolerance, score_tolerance = _rounding(
            len(keys), reference_probability_decimals
        )
        reported_error = (
            abs(_number(actual[qid]["score"], "score") - _number(expected[qid]["score"], "score"))
            if ak == "score"
            else None
        )
        questions.append(
            {
                "question_id": qid,
                "kind": ak,
                "reference_rounding": {
                    "probability_decimals": reference_probability_decimals,
                    "per_probability_half_unit": float(half_unit),
                    "normalization_tolerance": float(mass_tolerance)
                    if ak != "noul"
                    else float(_NORMALIZATION_TOLERANCE),
                    "score_consistency_tolerance": float(score_tolerance)
                    if ak == "score"
                    else None,
                    "probability_sum": float(sum(ev)),
                },
                "reported_score_error": float(reported_error)
                if reported_error is not None
                else None,
                "normalized_reported_score_error": float(reported_error / (len(keys) - 1))
                if reported_error is not None
                else None,
                "passed": probability_passed
                and (not policy["require_argmax_match"] or not changed),
                "probability_passed": probability_passed,
                "max_probability_error": float(max(errors)),
                "total_variation": float(sum(errors) / 2),
                "argmax_changed": changed,
                "choice_changed": changed if ak == "choice" else None,
                "noul_threshold_flip": changed if ak == "noul" else None,
                "expected_score_error": float(abs(ascore - escore)) if ak == "score" else None,
                "normalized_expected_score_error": float(abs(ascore - escore) / (len(keys) - 1))
                if ak == "score"
                else None,
            }
        )
    failed = [q["question_id"] for q in questions if not q["passed"]]
    return {
        "passed": not failed,
        "question_count": len(questions),
        "failed_questions": failed,
        "argmax_flip_count": sum(q["argmax_changed"] for q in questions),
        "max_probability_error": max(q["max_probability_error"] for q in questions),
        "questions": questions,
    }
