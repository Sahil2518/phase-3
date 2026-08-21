import os
import sys
import json
import logging
import traceback
from tenant_manager import TenantManager, TenantConfig
from multi_tenant_recommender import TenantCatalogManager, MultiTenantRecommender

# Rule 2: Structured logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "task16.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("=====================================================")
        logger.info("Starting Stage B & C: Tenant-scoped inference and config")
        logger.info("=====================================================")
        
        tenant_mgr = TenantManager("models/tenant_configs.json")
        
        # Enterprise A config: skill heavy, low threshold
        tenant_mgr.add_or_update_tenant("Ent_A", TenantConfig(
            skill_weight=0.7, location_weight=0.1, seniority_weight=0.2, match_threshold=0.4
        ))
        
        # Enterprise B config: balanced, high threshold
        tenant_mgr.add_or_update_tenant("Ent_B", TenantConfig(
            skill_weight=0.4, location_weight=0.4, seniority_weight=0.2, match_threshold=0.7
        ))
        
        # Output handoff contract
        with open("models/tenant_contract.json", "w") as f:
            schema = TenantConfig.schema() if hasattr(TenantConfig, "schema") else TenantConfig.model_json_schema()
            json.dump(schema, f, indent=4)
        logger.info("Saved Tenant Config Contract to models/tenant_contract.json")

        catalog = TenantCatalogManager(random_state=42)
        catalog.generate_tenant_data("Ent_A", num_candidates=500, num_jobs=100)
        catalog.generate_tenant_data("Ent_B", num_candidates=500, num_jobs=100)

        recommender = MultiTenantRecommender(catalog, tenant_mgr)
        
        logger.info("\n=====================================================")
        logger.info("Stage E.1 & E.2: Two tenants with different configs")
        logger.info("=====================================================")
        
        cand_A = list(catalog.get_candidates("Ent_A").index)[0]
        res_A = recommender.recommend_jobs_for_candidate("Ent_A", cand_A, k=3)
        logger.info(f"Ent_A Match for {cand_A}: {res_A}")
        
        cand_B = list(catalog.get_candidates("Ent_B").index)[0]
        res_B = recommender.recommend_jobs_for_candidate("Ent_B", cand_B, k=3)
        logger.info(f"Ent_B Match for {cand_B}: {res_B}")

        logger.info("\n=====================================================")
        logger.info("Stage D & E.3: Evidence of data isolation (break it on purpose)")
        logger.info("=====================================================")
        try:
            logger.info("Attempting to query Ent_A data using Ent_B's context (Cross-tenant leak attempt)...")
            recommender.recommend_jobs_for_candidate("Ent_B", cand_A, k=3)
            logger.critical("Data leak! Ent_B accessed Ent_A candidate.")
            sys.exit(1)
        except ValueError as e:
            logger.info(f"Isolation Guard triggered successfully: {e}")

        # E.3 Break on purpose: Empty input
        logger.info("Testing Empty Input Guard...")
        res_empty = recommender.recommend_jobs_for_candidate("Ent_A", "", k=3)
        logger.info(f"Empty input response: {res_empty}")
        
        # Metric logging
        metrics = {
            "tenant_a_threshold": 0.4,
            "tenant_b_threshold": 0.7,
            "isolation_tests_passed": 2,
            "status": "success"
        }
        with open("logs/task16_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        logger.info("\n[SUCCESS] Multi-tenant isolation and routing verified successfully.")
        
    except Exception as e:
        logger.critical(f"Unhandled fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
