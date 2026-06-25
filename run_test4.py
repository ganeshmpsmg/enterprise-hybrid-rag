import uvicorn 
 
config = uvicorn.Config("test_app:app", host="127.0.0.1", port=8000, lifespan="off") 
server = uvicorn.Server(config) 
print("START") 
server.run() 
print("STOP") 
print("should_exit =", server.should_exit) 
