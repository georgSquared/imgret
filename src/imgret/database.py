import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SQLALCHEMY_DATABASE_URL = f"sqlite:///{ROOT_DIR}/db/imgret.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=True, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    return db


class ModelManager:
    def __init__(self, model):
        self.Model = model
        self.db = get_db()

    def create(self, **kwargs):
        db_obj = self.Model(**kwargs)

        self.db.add(db_obj)
        self.db.flush()
        self.db.refresh(db_obj)

        return db_obj

    def get(self, obj_id):
        if not obj_id:
            return None

        return self.db.query(self.Model).filter(self.Model.id == obj_id).first()

    def filter(self, *args, **kwargs):
        return self.db.query(self.Model).filter(*args, **kwargs)

    def all(self):
        return self.db.query(self.Model).all()
