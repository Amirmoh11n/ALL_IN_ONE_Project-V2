from fastapi.testclient import TestClient
from webapplication.backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get('/api/health')
    assert response.status_code == 200
    body = response.json()
    assert 'model_ready' in body
    assert body['classes'] == ['glioma', 'meningioma', 'notumor', 'pituitary']
