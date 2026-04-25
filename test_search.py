from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

resp = client.post("/search", data={"question": "What is the refund policy?"})
print('STATUS', resp.status_code)
try:
    print(resp.json())
except Exception as e:
    print('Failed to decode JSON:', e)
    print(resp.text)
