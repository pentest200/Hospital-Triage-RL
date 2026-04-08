import os
from fastapi import FastAPI
from tasks.easy import make_easy_env

app = FastAPI()

# Use the easy task factory to create a properly initialized environment
env = make_easy_env(seed=42)

@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv ER Simulation is Running! 🚑"}

@app.get("/reset")
def get_reset():
    return reset_env()

@app.post("/reset")
def reset_env():
    obs = env.reset()
    return obs.model_dump()

@app.get("/state")
def get_state():
    return env.state()

def main():
    import uvicorn
    # Hugging Face Spaces explicitly require port 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
