import bson
from typing import Any
from pydantic import BaseModel


def validate_object_id(v: Any) -> bson.ObjectId:
    if isinstance(v, bson.ObjectId):
        return v
    if bson.ObjectId.is_valid(v):
        return bson.ObjectId(v)
    raise ValueError("Invalid ObjectId")


class MONGO(BaseModel):
    url: str
    dbname: str


class OPENAI(BaseModel):
    api_key: str
    base_url: str
    model: str
