import json

import numpy as np
from sklearn.metrics import pairwise_distances
from sqlalchemy import Column, Integer, Text, or_, ForeignKey
from sqlalchemy.orm import relationship

from imgret.database import Base, ModelManager
from imgret.utils import NumpyArrayEncoder, get_image_features


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    raw_features = Column(Text)
    file_path = Column(Text)
    file_name = Column(Text, index=True)

    ground_truth = relationship("GroundTruth", back_populates="image", lazy="subquery")

    @classmethod
    @property
    def manager(self):
        return ModelManager(self)

    @classmethod
    def add(cls, features=None, file_path=None, file_name=None):
        raw_features = json.dumps(features, cls=NumpyArrayEncoder)

        cls.manager.create(
            raw_features=raw_features, file_path=file_path, file_name=file_name
        )

    @classmethod
    def add_or_update(cls, features=None, file_path=None, file_name=None):
        raw_features = json.dumps(features, cls=NumpyArrayEncoder)

        existing_image = cls.manager.filter(
            or_(
                Image.file_name == file_name,
            )
        ).first()

        if existing_image:
            existing_image.raw_features = raw_features
            existing_image.file_path = file_path
            existing_image.file_name = file_name

            existing_image.flush()

            return existing_image

        cls.manager.create(
            raw_features=raw_features, file_path=file_path, file_name=file_name
        )

    @classmethod
    def get_relevant(cls, query_image, k=5, distance="sqeuclidean"):
        query_features = get_image_features(query_image)
        results = []

        for db_image in cls.manager.all():
            db_features = db_image.features

            if db_features.shape[0] > query_features.shape[0]:
                query_features.resize(db_features.shape)
            else:
                db_features.resize(query_features.shape)

            distance_array = pairwise_distances(
                query_features.reshape(1, -1),
                db_features.reshape(1, -1),
                metric=distance,
            )
            query_dist = distance_array.item()

            if len(results) < k:
                results.append(
                    {
                        "id": db_image.id,
                        "file_path": db_image.file_path,
                        "file_name": db_image.file_name,
                        "distance": query_dist,
                    }
                )
            else:
                if query_dist < results[-1]["distance"]:
                    results[-1] = {
                        "id": db_image.id,
                        "file_path": db_image.file_path,
                        "file_name": db_image.file_name,
                        "distance": query_dist,
                    }

            results = sorted(results, key=lambda d: d["distance"])

        return results

    def flush(self):
        db = self.__class__.manager.db
        db.add(self)
        db.flush()
        db.refresh(self)

        return self

    @property
    def features(self):
        feature_list = json.loads(self.raw_features)
        return np.array(feature_list)


class GroundTruth(Base):
    __tablename__ = "images_ground_truth"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey(Image.id, ondelete="CASCADE"), index=True)
    file_name = Column(Text)

    image = relationship("Image", back_populates="ground_truth")

    @classmethod
    @property
    def manager(self):
        return ModelManager(self)
