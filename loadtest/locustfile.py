from locust import HttpUser, task
from datetime import datetime, timedelta, timezone
import time


class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        prompt = "Hi, my name is"
        with self.client.post(
            "/",
            json={"prompt": prompt},
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure("Got wrong response")
                print("Got the wrong response!", response)
                print("Got the wrong response!", response.status_code)
                print("Got the wrong response!", response.text)
            else:
                output_json = response.json()

        time.sleep(1)
