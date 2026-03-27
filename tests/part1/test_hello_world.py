"""Deliverable verification: successful 'Hello World' response from the model endpoint."""
import pytest
from src.part1.engines import get_client


@pytest.mark.integration
def test_hello_world(default_config_path):
    """Send a Hello World prompt and assert a non-empty response is returned."""
    client, model = get_client(default_config_path)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say exactly: Hello World"}],
        max_tokens=32,
    )
    reply = response.choices[0].message.content
    assert reply and len(reply.strip()) > 0, "Expected a non-empty response"
    print(f"\nModel replied: {reply}")
