# database.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# mysql_address = os.getenv()


# DATABASE_URL = "172.19.0.3+"
MYSQL_HOST = os.getenv("MYSQL_HOST")  or "localhost"
MYSQL_PORT = os.getenv("MYSQL_PORT") or 3306
MYSQL_USER = os.getenv("MYSQL_USER") or "sampleuser"
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD") or "samplepassword"
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE") or "uis_summer"

# DATABASE_URL = "mysql+mysqldb://sampleuser:samplepassword@172.19.0.2:3306/uis_summer"
DATABASE_URL = f"mysql+mysqldb://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"



db_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()
