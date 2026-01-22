import random

from locust import HttpUser, between, task


class FraudDetectionUser(HttpUser):
    """
    Simulates users interacting with the Credit Card Fraud Detection API.
    """

    # Wait between 1 and 5 seconds between tasks to simulate human behavior
    wait_time = between(1, 5)

    @task(1)
    def get_root(self) -> None:
        """
        Simulates a user checking if the API is active.
        """
        self.client.get("/")

    @task(5)
    def predict_fraud(self) -> None:
        dummy_features = [random.uniform(-2, 2) for _ in range(28)]
        payload = {"features": dummy_features}

        # ADD catch_response=True here
        with self.client.post(
            "/predict",
            json=payload,
            name="/predict",
            catch_response=True,  # This is the required fix
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")
