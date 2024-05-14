from jwt import encode, decode
from fastapi import HTTPException

def create_token(data: dict) -> str:
    token: str = encode(payload=data, key="cviasuyedgfias", algorithm="HS256")
    return token

def validate_token(token: str) -> dict:
    try:
        data: dict = decode(token, key="cviasuyedgfias", algorithms=['HS256'])
        return data
    except Exception as e:
        raise HTTPException(status_code=403, detail="Credenciales invalidas")
