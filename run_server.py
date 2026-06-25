from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True}

if __name__ == "__main__":
    print("BEFORE SERVER")

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )

    server = uvicorn.Server(config)
    server.run()

    print("AFTER SERVER")