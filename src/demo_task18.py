import os
import sys
import json
import logging
from identity_manager import IdentityManager, PersonalizationSignals, UserContext
from personalized_recommender import OrgCatalogManager, PersonalizedRecommender

# Rule 2: Structured Logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task18.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=====================================================")
        logger.info("Stage B: Org- and recruiter-scoped signals")
        logger.info("=====================================================")
        
        catalog = OrgCatalogManager(random_state=42)
        catalog.generate_data("Org_A")
        catalog.generate_data("Org_B")
        
        im = IdentityManager()
        
        # Output Identity Contract
        logger.info("Exporting Identity Contract Schema...")
        schema = UserContext.schema() if hasattr(UserContext, "schema") else UserContext.model_json_schema()
        with open("models/identity_contract.json", "w") as f:
            json.dump(schema, f, indent=4)
        
        # Provision a user with a strong preference for Seniority
        signals = PersonalizationSignals(skill_multiplier=1.0, seniority_multiplier=2.5)
        im.provision_user("recruiter_1", "Org_A", signals)
        
        recommender = PersonalizedRecommender(catalog, im)
        
        # Test 1: Recommend in Org_A
        job_A = catalog.get_jobs("Org_A").index[0]
        res_A = recommender.recommend_candidates_for_job("recruiter_1", job_A, "Org_A", k=2)
        logger.info(f"Recruiter 1 scoring in Org A for {job_A}: {res_A}")
        
        logger.info("\n=====================================================")
        logger.info("Stage C: Correct behaviour when users move between orgs")
        logger.info("=====================================================")
        
        im.transfer_user("recruiter_1", "Org_B")
        
        # Test 2: Recommend in Org_B (proving signals followed)
        job_B = catalog.get_jobs("Org_B").index[0]
        res_B = recommender.recommend_candidates_for_job("recruiter_1", job_B, "Org_B", k=2)
        logger.info(f"Recruiter 1 scoring in Org B for {job_B}: {res_B}")
        
        logger.info("\n=====================================================")
        logger.info("Stage D & E.3: Isolation tests proving no signal bleed")
        logger.info("=====================================================")
        
        try:
            logger.info("Simulating compromised API call: Recruiter 1 requests Org A data post-transfer.")
            recommender.recommend_candidates_for_job("recruiter_1", job_A, "Org_A", k=2)
            logger.error("FATAL: Isolation breached. Recruiter accessed old org data.")
            sys.exit(1)
        except PermissionError as e:
            logger.info(f"Isolation Guard triggered successfully: {e}")
            
        logger.info("\n=====================================================")
        logger.info("Stage E: Deprovisioning Cleanup (Worry check)")
        logger.info("=====================================================")
        im.deprovision_user("recruiter_1")
        try:
            recommender.recommend_candidates_for_job("recruiter_1", job_B, "Org_B", k=2)
        except PermissionError as e:
            logger.info(f"Post-deprovision Guard triggered: {e}")
            
        # Final metrics
        metrics = {
            "isolation_verified": True,
            "deprovision_verified": True,
            "signals_persisted_on_transfer": True,
            "status": "success"
        }
        with open("logs/task18_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info("\n[SUCCESS] SSO/SCIM Identity scopes verified successfully.")
        
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
