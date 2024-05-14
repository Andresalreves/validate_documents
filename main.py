from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from config.database import engine, Base
from middlewares.error_handler import ErrorHandler
from routers.user import user_router
from routers.ValidateDocuments import ValidateDocuments_router

app = FastAPI()
app.title = "API reconocimiento facial"
app.version = "0.0.1"
#app.debug = True

app.add_middleware(ErrorHandler)
app.include_router(user_router)
app.include_router(ValidateDocuments_router)


Base.metadata.create_all(bind=engine)

@app.get('/', tags=['home'])
def message():
    return HTMLResponse('<div style=""><h1>Bienvenido a la api de validacion de documentos.</h1></div>')
