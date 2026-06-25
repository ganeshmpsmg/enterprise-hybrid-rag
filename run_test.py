from fastapi import FastAPI 
import uvicorn 
 
app = FastAPI() 
 
@app.get("/") 
def root(): 
    return {"status":"ok"} 
 
if __name__ == "__main__": 
    print("STARTING SERVER") 
    uvicorn.run(app, host="127.0.0.1", port=8000) 
    print("SERVER STOPPED") 
