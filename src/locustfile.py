from locust import HttpUser, task, between
import random

class InferenceUser(HttpUser):
    # Short wait time to simulate heavy concurrent load
    wait_time = between(0.01, 0.05)
    
    @task
    def predict_score(self):
        # Generate some random payload
        payload = {
            "skill_score": random.uniform(0.1, 1.0),
            "experience_years": random.randint(1, 15)
        }
        
        # We use name="/predict" so Locust groups the requests properly in the UI
        with self.client.post("/predict", json=payload, name="/predict", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 503:
                # We expect 503 during our failure test, we can mark it as success 
                # so the test doesn't look completely failed, but let's leave it as failure 
                # to see it in the error rate.
                response.failure(f"Service Unavailable: {response.text}")
            else:
                response.failure(f"Unexpected status: {response.status_code}")
