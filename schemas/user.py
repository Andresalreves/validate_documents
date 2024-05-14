from pydantic import BaseModel,Field,EmailStr


class User(BaseModel):
    email:EmailStr
    password:str = Field(min_length=8, max_length=20, pattern=r"^[a-zA-Z0-9_\-@#]+$")

    class Config:
        schema_extra = {
            "example": {
                "email": "prueba@gmail.com",
                "password": "imvhAdo5upVdTpCMhVV2"
            }
        }