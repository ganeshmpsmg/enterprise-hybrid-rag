import uvicorn 
 
if __name__ == "__main__": 
    print("START") 
    uvicorn.run("test_app:app", host="127.0.0.1", port=8000, reload=False, workers=1, lifespan="off") 
    print("STOP") 
