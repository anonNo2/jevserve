"""Declared model budgets; populated only from a verified model manifest."""

from typing import Annotated, Literal

from pydantic import Field

from jevserve.contracts.requests import Contract, DecisionRequest


class Capabilities(Contract):
    question_types: Annotated[list[Literal["choice", "noul", "score"]], Field(min_length=1)]
    max_choice_candidates: Annotated[int, Field(ge=1)]
    max_score_levels: Annotated[int, Field(ge=2)]
    max_tokens_per_candidate: Annotated[int, Field(ge=1)]
    max_candidate_sequences: Annotated[int, Field(ge=1)]
    max_total_input_tokens: Annotated[int, Field(ge=1)]
    max_padded_input_tokens: Annotated[int, Field(ge=1)] | None = None
    min_choice_candidates: Annotated[int, Field(ge=1)] = 1
    noul_candidate_cost: Annotated[int, Field(ge=1, le=2)] = 1

    def validate_request(self, request: DecisionRequest) -> None:
        count = 0
        for question in request.questions.values():
            if question.type not in self.question_types:
                raise ValueError("unsupported question type")
            if question.type == "choice":
                count += len(question.criteria)
                if (
                    not self.min_choice_candidates
                    <= len(question.criteria)
                    <= self.max_choice_candidates
                ):
                    raise ValueError("choice candidate limit exceeded")
            elif question.type == "score":
                count += len(question.criteria)
                if len(question.criteria) > self.max_score_levels:
                    raise ValueError("score level limit exceeded")
            else:
                count += self.noul_candidate_cost
        if count > self.max_candidate_sequences:
            raise ValueError("candidate work budget exceeded")

    def validate_token_lengths(self, lengths: list[int]) -> None:
        if not lengths or len(lengths) > self.max_candidate_sequences:
            raise ValueError("invalid number of candidate sequences")
        if any(type(n) is not int or not 1 <= n <= self.max_tokens_per_candidate for n in lengths):
            raise ValueError("context length exceeded or invalid")
        if (
            self.max_padded_input_tokens is not None
            and len(lengths) * max(lengths) > self.max_padded_input_tokens
        ):
            raise ValueError("padded token work budget exceeded")
        if sum(lengths) > self.max_total_input_tokens:
            raise ValueError("total token work budget exceeded")
