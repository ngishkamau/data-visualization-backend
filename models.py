from datetime import date, datetime
# from turtle import update
from xmlrpc.client import DateTime

from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from database import Base


class FileCollection(Base):
    __tablename__ = 'file_collection'

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(100))
    filesize = Column(String(100))
    user_id = Column(Integer, ForeignKey('users.id'))

    uploaded_by = relationship("User", back_populates="file_collection")
    # file_requests = relationship('FileRequest', back_populates="file_id_f")

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100))
    email = Column(String(100))
    password = Column(String(100))

    file_collection = relationship('FileCollection', back_populates="uploaded_by")

class FileRequest(Base):
    __tablename__ = 'file_requests'

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer)
    sender = Column(Integer)
    reciever = Column(Integer)
    status = Column(Integer)

    # file_id_f = relationship('FileCollection', back_populates="file_requests")

class Rig(Base):
    __tablename__ = 'rigs'

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey('users.id'))
    latitude = Column(String(100))
    longitude = Column(String(100))
    name = Column(String(100))

class Model(Base):
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey('file_collection.id'))
    owner_id = Column(Integer, ForeignKey('users.id'))
    rig_id = Column(Integer, ForeignKey('rigs.id'))
    name = Column(String(100))

class ModelRequest(Base):
    __tablename__ = 'model_requests'

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    sender = Column(Integer)
    reciever = Column(Integer)
    status = Column(Integer)

class RigRequest(Base):
    __tablename__ = 'rig_requests'

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer)
    sender = Column(Integer)
    reciever = Column(Integer)
    status = Column(Integer)
