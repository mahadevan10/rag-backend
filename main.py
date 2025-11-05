"""Entry point for running the Vegah backend locally."""
from __future__ import annotations

from backend import create_app

app = create_app()

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
