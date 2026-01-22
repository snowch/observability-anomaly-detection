import requests
import time
import random
from datetime import datetime, timedelta
import json
import os

class LoadGenerator:
    """
    Generate realistic observability data with normal and anomalous patterns.
    """
    def __init__(self, target_url, normal_rps=10, anomaly_probability=0.05):
        self.target_url = target_url
        self.normal_rps = normal_rps
        self.anomaly_probability = anomaly_probability
        self.session = requests.Session()

    def generate_normal_traffic(self):
        """Generate normal user traffic patterns."""
        patterns = [
            # Pattern 1: User browsing
            lambda: self.session.get(f"{self.target_url}/api/users/{random.randint(1, 1000)}"),

            # Pattern 2: Search
            lambda: self.session.get(f"{self.target_url}/api/search?q=product"),

            # Pattern 3: Checkout (normal)
            lambda: self.session.post(f"{self.target_url}/api/checkout", json={"cart_id": random.randint(1, 100)}),
        ]

        pattern = random.choice(patterns)
        try:
            response = pattern()
            print(f"[NORMAL] {response.status_code} - {response.url}")
        except Exception as e:
            print(f"[ERROR] {str(e)}")

    def generate_anomaly_scenario(self):
        """Generate specific anomaly scenarios."""
        scenarios = [
            self.scenario_deployment_memory_leak,
            self.scenario_database_connection_pool_exhaustion,
            self.scenario_cache_invalidation_storm,
            self.scenario_slow_query_cascade,
        ]

        scenario = random.choice(scenarios)
        print(f"\n[ANOMALY] Starting scenario: {scenario.__name__}")
        scenario()

    def scenario_deployment_memory_leak(self):
        """
        Simulate memory leak after deployment (gradual degradation).

        Sequence:
        1. Deployment completes (config change)
        2. Memory usage gradually increases
        3. GC pressure increases
        4. Query latency spikes
        5. Connection pool exhaustion
        """
        print("  -> Simulating deployment with memory leak")

        # Generate increasing load over 5 minutes
        for i in range(60):
            try:
                # Each request allocates more memory in the service
                response = self.session.post(
                    f"{self.target_url}/api/checkout",
                    json={"trigger_memory_leak": True}
                )
                print(f"  -> Minute {i//12}: Memory pressure increasing")
                time.sleep(5)
            except Exception as e:
                print(f"  -> Service degraded: {str(e)}")
                break

    def scenario_database_connection_pool_exhaustion(self):
        """
        Simulate DB connection pool exhaustion.

        Sequence:
        1. Spike in concurrent requests
        2. Slow queries hold connections
        3. Pool exhausted
        4. New requests timeout
        """
        print("  -> Simulating DB connection pool exhaustion")

        import concurrent.futures

        def slow_request():
            try:
                response = self.session.get(
                    f"{self.target_url}/api/users/{random.randint(1, 1000)}",
                    timeout=10
                )
                return response.status_code
            except:
                return 500

        # Flood with 100 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(slow_request) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        print(f"  -> Results: {results.count(200)} success, {results.count(500)} failed")

    def scenario_cache_invalidation_storm(self):
        """
        Simulate cache invalidation causing DB overload.

        Sequence:
        1. Cache gets invalidated (deployment or manual flush)
        2. All requests hit database
        3. DB overloaded
        4. Latency spikes
        """
        print("  -> Simulating cache invalidation storm")

        # Trigger cache invalidation
        self.session.post(f"{self.target_url}/admin/cache/flush")

        # Generate high read traffic (all cache misses)
        for i in range(100):
            try:
                response = self.session.get(
                    f"{self.target_url}/api/users/{random.randint(1, 10000)}"
                )
                if i % 10 == 0:
                    print(f"  -> Cache miss {i}: {response.elapsed.total_seconds():.2f}s")
            except:
                pass
            time.sleep(0.1)

    def scenario_slow_query_cascade(self):
        """
        Simulate slow query causing cascading failure.

        Sequence:
        1. One slow query blocks resources
        2. Other queries queue up
        3. Thread pool exhaustion
        4. Service unresponsive
        """
        print("  -> Simulating slow query cascade")

        # Trigger slow query
        self.session.post(
            f"{self.target_url}/api/analytics",
            json={"trigger_slow_query": True}
        )

        time.sleep(30)  # Let cascade develop

    def run(self, duration_minutes=60):
        """
        Run load generator for specified duration.

        Args:
            duration_minutes: How long to generate traffic
        """
        end_time = datetime.now() + timedelta(minutes=duration_minutes)

        print(f"Starting load generator (RPS: {self.normal_rps}, anomaly prob: {self.anomaly_probability})")
        print(f"Will run until: {end_time}")

        request_count = 0
        anomaly_count = 0

        while datetime.now() < end_time:
            # Decide if this cycle is normal or anomaly
            if random.random() < self.anomaly_probability:
                self.generate_anomaly_scenario()
                anomaly_count += 1
            else:
                # Generate normal traffic at target RPS
                for _ in range(self.normal_rps):
                    self.generate_normal_traffic()
                    time.sleep(1.0 / self.normal_rps)
                    request_count += 1

            # Print progress every minute
            if request_count % (self.normal_rps * 60) == 0:
                print(f"\n[PROGRESS] Generated {request_count} requests, {anomaly_count} anomaly scenarios")

if __name__ == '__main__':
    target_url = os.getenv('TARGET_URL', 'http://web-api:8000')
    normal_rps = int(os.getenv('NORMAL_RPS', 10))
    anomaly_probability = float(os.getenv('ANOMALY_PROBABILITY', 0.05))

    generator = LoadGenerator(target_url, normal_rps, anomaly_probability)
    generator.run(duration_minutes=120)  # Run for 2 hours
