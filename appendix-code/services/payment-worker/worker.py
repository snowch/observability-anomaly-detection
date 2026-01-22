import os
import time
import random
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp":"%(asctime)s","service":"payment-worker","level":"%(levelname)s","message":"%(message)s"}'
)
logger = logging.getLogger(__name__)

def process_payment(payment_id):
    """Simulate payment processing with logging."""
    start_time = time.time()

    # Simulate processing time
    processing_time = random.uniform(0.1, 2.0)
    time.sleep(processing_time)

    # Simulate occasional failures
    if random.random() < 0.05:
        logger.error(f"Payment {payment_id} failed: timeout")
        return False

    duration = (time.time() - start_time) * 1000
    logger.info(f"Payment {payment_id} processed successfully in {duration:.2f}ms")
    return True

def main():
    """Simple worker loop for demo purposes."""
    logger.info("Payment worker started")

    while True:
        # Simulate receiving jobs from queue
        payment_id = random.randint(10000, 99999)
        process_payment(payment_id)

        # Wait before next job
        time.sleep(random.uniform(1, 5))

if __name__ == '__main__':
    main()
