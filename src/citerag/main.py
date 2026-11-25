from fastapi import FastAPI

from citerag import __version__

app = FastAPI(
    title="citerag",
    version=__version__,
    description="Document Q&A API with grounded, cited answers.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
