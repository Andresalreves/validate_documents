from fastapi import APIRouter
from config.database import Session
from utils.jwt_manager import create_token
from utils.hashing import verify_password
from fastapi.responses import JSONResponse
from schemas.user import User
from services.User import UserService
from fastapi.encoders import jsonable_encoder


user_router = APIRouter()


@user_router.post('/CreateUser', tags=['Users'], response_model=dict, status_code=201)
def create_user(user: User) -> dict:
    db = Session()
    UserService(db).create_User(user)
    return JSONResponse(status_code=201, content={"code":201,"message": "Se ha registrado el usuario"})

@user_router.post('/auth', tags=['auth'])
def login(user: User):
    db = Session()
    AuthUser = UserService(db).get_UserByEmail(user.email)
    if bool(AuthUser):
        if verify_password(user.password,AuthUser.password):
            token: str = create_token(jsonable_encoder(AuthUser))
            return JSONResponse(status_code=200, content=token)
        else:
            return JSONResponse(status_code=403, content={'code':403,"message":"Password incorrecto"})
    else:
        return JSONResponse(status_code=404, content={"code":404,"message":"Usuario no registrado."})