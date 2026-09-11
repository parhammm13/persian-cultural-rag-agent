from fastapi.testclient import (
    TestClient,
)

from src.api.main import create_app
from src.api.request_context import (
    REQUEST_ID_HEADER,
)


class ExplodingPipeline:
    def run(
        self,
        query: str,
    ):
        raise RuntimeError(
            "OPENROUTER_API_KEY=super-secret-value"
        )


def test_validation_error_uses_public_error_contract():
    app = create_app(
        lambda: ExplodingPipeline()
    )

    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "query": "   ",
            },
        )

    assert response.status_code == 422

    body = response.json()

    assert body["error"] == (
        "validation_error"
    )

    assert body["request_id"] == (
        response.headers[
            REQUEST_ID_HEADER
        ]
    )


def test_unhandled_exception_is_not_leaked_to_client():
    app = create_app(
        lambda: ExplodingPipeline()
    )

    with TestClient(
        app,
        raise_server_exceptions=False,
    ) as client:
        response = client.post(
            "/query",
            json={
                "query": "ارگ بم کجاست؟",
            },
        )

    assert response.status_code == 500

    assert response.json() == {
        "error": (
            "internal_server_error"
        ),
        "message": (
            "An internal server error occurred."
        ),
        "request_id": response.headers[
            REQUEST_ID_HEADER
        ],
    }

    assert (
        "super-secret-value"
        not in response.text
    )


def test_not_found_uses_http_error_contract():
    app = create_app(
        lambda: ExplodingPipeline()
    )

    with TestClient(app) as client:
        response = client.get(
            "/does-not-exist"
        )

    assert response.status_code == 404

    assert response.json() == {
        "error": "http_error",
        "message": "Not Found",
        "request_id": response.headers[
            REQUEST_ID_HEADER
        ],
    }