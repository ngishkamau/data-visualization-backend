from ast import For
from typing import List, Optional, Union

from pydantic import BaseModel, EmailStr


class FileCollection(BaseModel):
    id: Union[int, None] = None
    filename: str
    filesize: str

    class Config():
        orm_mode = True

class User(BaseModel):
    id: Union[int, None] = None
    name: str
    email: EmailStr
    password: str

class ShowUser(BaseModel):
    id: int
    name: str
    email: EmailStr
    file_collection : List[FileCollection] =[]
    class Config():
        orm_mode = True

class ShowFileCollection(BaseModel):
    id: int
    filename: str
    filesize: str
    uploaded_by: ShowUser

    class Config():
        orm_mode = True

class FileRequest(BaseModel):
    id: int
    fileid: int
    sender: int
    reciever: int
    status: int

class FileRequestSend(BaseModel):
    file_id: int
    file_owner: int

class Login(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None

class FLModel(BaseModel):
    task: str
    epochs: int
    model: str
    dataset: str
    appr: str