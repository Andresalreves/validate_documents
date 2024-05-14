from fastapi.security import HTTPBearer
from fastapi import Request, HTTPException
from utils.jwt_manager import create_token, validate_token
from services.User import UserService
from config.database import Session

class JWTBearer(HTTPBearer):
    async def __call__(self, request: Request):
        auth = await super().__call__(request)
        data = validate_token(auth.credentials)
        db = Session()
        AuthUser = UserService(db).get_User(data["id"])
        if data['email'] != AuthUser.email:
            raise HTTPException(status_code=403, detail="Credenciales invalidas")