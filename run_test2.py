from fastapi import FastAPI 
import uvicorn 
import time 
 
app = FastAPI() 
 
@app.get("/") 
def root(): 
    return {"status":"ok"} 
 
print("BEFORE UVICORN") 
uvicorn.run(app, host="127.0.0.1", port=8000) 
print("AFTER UVICORN") 
time.sleep(30) 
print("END") 
