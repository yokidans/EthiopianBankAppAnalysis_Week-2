import sys
import os
import importlib.util
import logging

project_root = os.path.abspath(os.path.dirname(__file__))
# Set up basic logging for debugging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

logger.info(f"__main__.py is executing.")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"sys.path: {sys.path}")

# --- Diagnostic for 'src.oracle_connector' package ---
# Check if the parent package 'src' is discoverable
if 'src' not in sys.modules:
    logger.warning("'src' not found in sys.modules. This might be part of the issue.")
    # Try a direct import of 'src' to see if it makes it discoverable
    try:
        import src
        logger.info("Successfully attempted 'import src'.")
    except ImportError as e:
        logger.error(f"Failed to import 'src' directly: {e}")

# Check if 'src.oracle_connector' is discoverable
oracle_connector_spec = importlib.util.find_spec('src.oracle_connector')
if oracle_connector_spec:
    logger.info(f"Found spec for 'src.oracle_connector': {oracle_connector_spec.origin}")
else:
    logger.error("Could not find spec for 'src.oracle_connector'.")
    logger.error("This means Python doesn't see 'src' as a package containing 'oracle_connector'.")
    logger.error("Double-check that C:\\Users\\tefer\\Desktop\\database is in sys.path and has a 'src' folder with __init__.py.")
    logger.error("Also check that 'src/oracle_connector' folder has __init__.py.")

# --- Diagnostic for 'review_data_loader' ---
# Check if the specific module 'review_data_loader' is discoverable within the package
review_loader_module_name = 'src.oracle_connector.review_data_loader'
review_loader_spec = importlib.util.find_spec(review_loader_module_name)
if review_loader_spec:
    logger.info(f"Found spec for '{review_loader_module_name}': {review_loader_spec.origin}")
else:
    logger.error(f"Could not find spec for '{review_loader_module_name}'.")
    logger.error(f"This indicates Python cannot locate the '{review_loader_module_name}' module.")
    logger.error(f"Ensure review_data_loader.py exists inside src/oracle_connector and has correct permissions.")


try:
    # This is your original import line
    from .review_data_loader import main
    logger.info("Successfully imported 'main' from '.review_data_loader'.")

    # If the import succeeds, call your main function
    main()
    logger.info("review_data_loader.main() executed successfully.")

except ImportError as e:
    logger.critical(f"FATAL ERROR: Failed to import main from review_data_loader. Error: {e}")
    logger.critical("This means the module or its contents cannot be found.")
except Exception as e:
    logger.critical(f"An unexpected error occurred during execution: {e}")