from fastapi import FastAPI

app = FastAPI("Automated ML Retraining Pipeline")

@app.post("/predict") 
async def predict():
    return 0