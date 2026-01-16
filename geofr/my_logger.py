from loguru import logger
import sys

# Configure log level (INFO is default)
logger.add("logs/app.log", rotation="100 MB", level="DEBUG")

# Export logger so other files can import it
__all__ = ["logger"]

# To only show warnings and above in terminal:
logger.remove()  # remove default handler
logger.add("logs/app.log", rotation="100 MB", level="DEBUG")  # file
logger.add(sys.stderr, level="WARNING")  # terminal

# Integrate with Hydra
import os
hydra_log_dir = os.getcwd()  # Hydra changes working directory
logger.add(f"{hydra_log_dir}/app.log", rotation="100 MB")
