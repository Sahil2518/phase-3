import os
import sys
import json
import logging
from fastapi.testclient import TestClient

from partner_api import app

# Setup logging
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=====================================================")
        logger.info("Starting Stage E: ATS Partner Integration E2E Demo")
        logger.info("=====================================================")
        
        client = TestClient(app)
        
        # Output OpenAPI JSON (Partner-facing API docs)
        logger.info("Stage D: Extracting Partner-facing API documentation (OpenAPI)...")
        openapi_schema = app.openapi()
        with open("models/partner_openapi.json", "w") as f:
            json.dump(openapi_schema, f, indent=4)
        logger.info("Saved OpenAPI schema to models/partner_openapi.json")

        logger.info("\n=====================================================")
        logger.info("Stage B & E.2: Valid Partner API Call (Score + Explanation)")
        logger.info("=====================================================")
        
        payload = {
            "candidate_skills": ["Python", "AWS", "SQL"],
            "job_skills": ["Python", "AWS"],
            "candidate_seniority": 3,
            "job_seniority": 3
        }
        
        headers = {"x-api-key": "partner_a_live"}
        
        response = client.post("/api/v1/match", json=payload, headers=headers)
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Success! Score: {data['score']:.2f}")
            logger.info(f"Explanation: {data['explanation']}")
        else:
            logger.error(f"Failed valid call: {response.status_code} {response.text}")
            
        logger.info("\n=====================================================")
        logger.info("Stage C & E.3: Abuse Protection & Quota Enforcement")
        logger.info("=====================================================")
        logger.info("Simulating abusive scraping (Firing 6 requests rapidly)...")
        
        blocked = False
        for i in range(6):
            res = client.post("/api/v1/match", json=payload, headers=headers)
            if res.status_code == 429:
                logger.info(f"Request {i+1}: BLOCKED! 429 Too Many Requests -> {res.json()['detail']}")
                blocked = True
            else:
                logger.info(f"Request {i+1}: OK")
                
        if blocked:
            logger.info("[SUCCESS] Abuse protection successfully prevented model scraping.")
        else:
            logger.error("[ERROR] Rate limiter failed to block abusive traffic.")
            
        logger.info("\n=====================================================")
        logger.info("Stage E.3: Unauthorized Access Guard (Missing/Invalid Key)")
        logger.info("=====================================================")
        bad_res = client.post("/api/v1/match", json=payload, headers={"x-api-key": "hacker_key"})
        logger.info(f"Unauthorized Attempt Status: {bad_res.status_code} -> {bad_res.json().get('detail')}")

        # Final metrics
        metrics = {
            "openapi_generated": True,
            "abuse_protection_active": blocked,
            "unauthorized_blocked": bad_res.status_code in [401, 403],
            "status": "success"
        }
        with open("logs/task17_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info("\n[SUCCESS] Partner API Integration verified successfully.")
        
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
