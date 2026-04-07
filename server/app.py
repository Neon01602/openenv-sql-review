"""server/app.py — OpenEnv server entry point.

Required by `openenv validate` local file structure check.
Re-exports `app` from app.main so that both of these work:
    uvicorn server.app:app        ← openenv validate / Dockerfile
    uvicorn app.main:app          ← direct local dev
    python  server/app.py         ← openenv serve
"""
from __future__ import annotations

import os
import sys

# Ensure the project root is on the path so `app.*` imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Re-export the real FastAPI application object
from app.main import app  # noqa: E402  ← this is what uvicorn needs


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