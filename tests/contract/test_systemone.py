import httpx2
import pytest
from conftest import FakeRuntime
from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul, RetryPolicy, Score

from jevserve.artifacts.catalog import DEFAULT_MODEL
from tests.contract.test_http import app_for


@pytest.mark.asyncio
async def test_frozen_official_sdk_against_asgi_service():
    app = app_for(FakeRuntime())
    async with (
        app.router.lifespan_context(app),
        AsyncTypeSafeClient(
            api_key="test-key",
            model=DEFAULT_MODEL,
            base_url="http://test",
            retry=RetryPolicy(max_retries=0),
            transport=httpx2.ASGITransport(app),
        ) as client,
    ):
        result = await client.system_one(
            state="A billing question",
            questions={
                "team": Choice(
                    instructions="Choose a team", criteria={"billing": None, "support": None}
                ),
                "urgent": Noul(instructions="Is this urgent?"),
                "priority": Score(instructions="How urgent?", criteria=["Low", "High"]),
            },
        )
        assert result.model == DEFAULT_MODEL
        assert result.usage.input_tokens > 0 and result.usage.output_tokens == 0
        assert result.choices["team"].choice == "billing"
        assert result.nouls["urgent"].noul == 0.5
        assert result.scores["priority"].score == 0.5


def test_pointer_score_confidence_preserves_author_distance_definition():
    from jevserve.contracts.responses import ScoreAnswer
    from jevserve.contracts.systemone import systemone_response

    answer = ScoreAnswer(
        type="score",
        score=1.0,
        legend={"0": "Low", "1": "Medium", "2": "High"},
        probabilities={"0": 0.25, "1": 0.5, "2": 0.25},
    )
    response = systemone_response(
        {"q": answer}, 10, model_id="jaredpalmer/kev-4b", pointer_confidence=True
    )
    assert response["model"] == "jaredpalmer/kev-4b"
    assert response["answers"]["q"]["confidence"] == 0.75
