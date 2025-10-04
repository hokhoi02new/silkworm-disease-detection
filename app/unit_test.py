from fastapi.testclient import TestClient
from app.app_api import app, startup_event
import pytest

@pytest.fixture(scope="session", autouse=True)
def load_models():
    import asyncio
    asyncio.run(startup_event())

client = TestClient(app)

TEST_VIDEO = "test/test_video.mp4"
TEST_IMAGE = "test/test_image.jpg"

def test_predict_video():
    with open(TEST_VIDEO, "rb") as f:
        response = client.post("/predict_video/",files={"file": ("test_video.mp4", f, "video/mp4")},)

    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 1000

def test_predict_image():
    with open(TEST_IMAGE, "rb") as f:
        response = client.post("/predict_image/",files={"file": ("test_image.jpg", f, "image/jpeg")},)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000


