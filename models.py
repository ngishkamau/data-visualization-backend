from database import Base
from datetime import datetime
from sqlalchemy.orm import relationship
from sqlalchemy import Column, ForeignKey, Integer, String, Enum, Boolean, Text, DateTime


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

class Dataset(Base):
    __tablename__ = 'datasets'

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey('users.id'))
    dataset = Column(String(100))
    description = Column(String(255))
    affiliation = Column(String(255))
    filetype = Column(Enum('image', 'audio', 'video', 'text', 'csv'))
    filepath = Column(Integer, ForeignKey('file_collection.id'))
    datatype = Column(Enum('CV', 'NLP', 'ASR'))

class DatasetPermission(Base):
    __tablename__ = 'dataset_permission'

    id = Column(Integer, primary_key=True, index=True)
    did = Column(Integer, ForeignKey('datasets.id')) 
    uid = Column(Integer, ForeignKey('users.id'))

class LearningModel(Base):
    __tablename__ = 'learning_models'

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey('users.id'))
    job_id = Column(String(100))
    task = Column(String(100))
    epochs = Column(Integer)
    model = Column(Enum('CNN', 'ResNet18', 'ResNet', '50', 'VGG19'))
    dataset = Column(Enum('MINIST', 'CIFAR10', 'CFIFAR100'))
    aggregation_approach = Column(Enum('FedSGD', 'FedAVG'))
    address = Column(String(50))
    port = Column(Integer)
    image_name = Column(String(100))
    container_id = Column(String(100))
    tensorboard_port = Column(Integer)
    tensorboard_image = Column(String(100))
    tensorboard_container = Column(String(100))
    global_model_port = Column(Integer)
    global_model_container = Column(String(100))
    link = Column(String(100))

class InternalMessage(Base):
    __tablename__ = 'internal_message'

    id = Column(Integer, primary_key=True, index=True)
    receiver = Column(Integer, ForeignKey('users.id'))
    sender = Column(Integer, ForeignKey('users.id'))
    title = Column(String(100))
    content = Column(Text)
    have_read = Column(Boolean, default=False)
    send_at = Column(DateTime, default=datetime.now())