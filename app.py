from src.state import initialize
from src.server import app, start_server

initialize()

if __name__ == "__main__":
    start_server()
