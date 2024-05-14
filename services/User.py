from models.user import User as UserModel
from schemas.user import User
from utils.hashing import hash_password

class UserService():
    
    def __init__(self, db) -> None:
        self.db = db

    def get_Users(self):
        result = self.db.query(UserModel).all()
        return result

    def get_User(self, id):
        result = self.db.query(UserModel).filter(UserModel.id == id).first()
        return result

    def create_User(self, user: User):
        hashed_password = hash_password(user.password)
        new_user = UserModel(email=user.email, password=hashed_password)
        self.db.add(new_user)
        self.db.commit()
        return

    def get_UserByEmail(self, email):
        result = self.db.query(UserModel).filter(UserModel.email == email).first()
        return result

    def update_User(self, id: int, data: User):
        User = self.db.query(UserModel).filter(UserModel.id == id).first()
        User.email = data.email
        User.password = hash_password(data.password)
        self.db.commit()
        return

    def delete_User(self, id: int):
       self.db.query(UserModel).filter(UserModel.id == id).delete()
       self.db.commit()
       return