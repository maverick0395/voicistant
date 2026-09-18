from fastapi import FastAPI

app = FastAPI(title="Voicistant")


@app.get("/health")
def health():
    return {"ok": True}
