"""
server/app.py — OpenEnv server entry point.

Required by `openenv validate` local file structure check.
The actual FastAPI app lives in app/main.py; this file bridges
the openenv serve / uv run server entry point to it.
"""
from __future__ import annotations

import os
import sys

# Ensure the project root is on the path so `app.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    """Start the SQL Query Review OpenEnv server."""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "7860")),
        workers=1,
    )


if __name__ == "__main__":
    main()
