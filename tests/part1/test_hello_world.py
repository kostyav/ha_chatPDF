"""Deliverable verification: successful 'Hello World' response from the model endpoint."""
import pytest
import yaml
from src.part1.engines import get_client
from .conftest import ENGINE_SERVER_FIXTURE


@pytest.mark.integration
def test_hello_world(default_config_path, request):
    """Send a Hello World prompt and assert a non-empty response is returned."""
    engine = yaml.safe_load(default_config_path.read_text())["engine"]
    request.getfixturevalue(ENGINE_SERVER_FIXTURE[engine])
    client, model = get_client(default_config_path)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly: Hello World"}],
        max_tokens=32,
    )
    reply = response.choices[0].message.content
    assert reply and len(reply.strip()) > 0, "Expected a non-empty response"
    assert "hello world" in reply.strip().lower(), f"Expected 'Hello World' in response, got: {reply!r}"
    print(f"\nModel replied: {reply}")
