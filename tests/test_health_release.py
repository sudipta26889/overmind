import pytest

pytestmark = pytest.mark.covers("boot")


@pytest.mark.django_db
def test_health_reports_the_running_release(client, monkeypatch):
    monkeypatch.setenv("RELEASE_SHA", "abc1234")

    assert client.get("/health").json() == {"status": "healthy", "release": "abc1234"}
