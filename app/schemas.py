from pydantic import BaseModel, EmailStr, Field
from typing import List, Dict, Optional



class ConvoCreate(BaseModel):
    #user_id: int
    image_id: int
    conversations: List[Dict]
    model_name: str
    model_type: str
    task: str
    feedback: str
    monetized: Optional[bool] = True
    enabled: Optional[bool] = True


class ImageCreate(BaseModel):
    sha256: str
    phash: str
    image_url: str
    image_path: str
    content_length: int


class ImgHashCheck(BaseModel):
    sha256: Optional[str] = None
    phash: Optional[str] = None
    content_length: int


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)

class SignupResponse(BaseModel):
    id: int
    email: EmailStr
