import os
from fastapi import FastAPI
from triage_env.environment import ERSimulationEnv

app = FastAPI()
env = ERSimulationEnv()

@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv ER Simulation is Running! 🚀"}

@app.get("/reset")
def get_reset():
    return reset_env()

@app.post("/reset")
def reset_env():
    obs = env.reset()
    # Pydantic V2 dump
    return obs.model_dump()

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces explicitly require port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
