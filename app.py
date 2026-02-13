import sys
from dotenv import load_dotenv
load_dotenv("config.env")

from src.logging_handler import logger
from src.state import initialize
from src.server import app, start_server

try:
    initialize()
except Exception as e:
    logger.critical(f"Failed to initialize: {e}")
    sys.exit(1)

if __name__ == "__main__":
    start_server()
