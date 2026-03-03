"""
Bulk Upload Routes for Elbiat

Supports:
1. Multi-image selection (up to 50 images)
2. Zip/tar.gz archive uploads (images only)
3. Image-Caption pairings (zip with images + captions.json)
4. Image-Instruction pairings (zip with images + instructions.json)
"""

import os
import json
import zipfile
import tarfile
import tempfile
import shutil
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from sqlalchemy import select

from app.db import get_db
#from app.auth import get_current_user
from app.deps import get_current_user

from app import models
from PIL import Image
from io import BytesIO

from app.routes.images import (
    _sha256_bytes,
    _ext_from_content_type,
    _new_image_fn,
    _resize_for_saving,
    _atomic_write,
    _compute_phash,
    IMAGES_DIR,
)


router = APIRouter(prefix="/bulk", tags=["bulk-upload"])

# Configuration
MAX_IMAGES_PER_UPLOAD = 50
MAX_ARCHIVE_SIZE_MB = 4000
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
ALLOWED_ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tar.gz", ".tgz"}

# Storage path - adjust to your setup
UPLOAD_BASE_PATH = os.environ.get("UPLOAD_PATH", "/home/ubuntu/workspace/elbiat/uploads")


# ============== Pydantic Models ==============

class BulkUploadResponse(BaseModel):
    success: bool
    message: str
    total_files: int
    successful: int
    failed: int
    image_ids: List[int]
    caption_ids: List[int] = []
    instruction_ids: List[int] = []
    errors: List[str]



class CaptionPairing(BaseModel):
    filename: str
    caption: str


class InstructionPairing(BaseModel):
    filename: str
    instruction: str
    response: Optional[str] = None


class ArchiveFormat(BaseModel):
    """Expected archive format documentation"""
    format_type: str
    description: str
    example_structure: dict


# ============== Helper Functions ==============

def is_allowed_image(filename: str) -> bool:
    """Check if file has allowed image extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def is_allowed_archive(filename: str) -> bool:
    """Check if file has allowed archive extension."""
    name_lower = filename.lower()
    return any(name_lower.endswith(ext) for ext in ALLOWED_ARCHIVE_EXTENSIONS)


def get_file_extension(filename: str) -> str:
    """Get normalized file extension."""
    return Path(filename).suffix.lower()


def validate_archive_contents(file_list: List[str], require_metadata: str = None) -> tuple[List[str], List[str]]:
    """
    Validate archive contents.
    Returns (valid_images, errors)
    """
    valid_images = []
    errors = []
    has_metadata = False
    
    for filepath in file_list:
        # Skip directories
        if filepath.endswith('/'):
            continue
        
        # Skip hidden files and __MACOSX
        basename = os.path.basename(filepath)
        if basename.startswith('.') or '__MACOSX' in filepath:
            continue
        
        # Check for metadata file
        if require_metadata and basename == require_metadata:
            has_metadata = True
            continue
        
        # Check if it's an allowed image
        if is_allowed_image(filepath):
            valid_images.append(filepath)
        else:
            errors.append(f"Invalid file type: {filepath}")
    
    if require_metadata and not has_metadata:
        errors.insert(0, f"Missing required metadata file: {require_metadata}")
    
    return valid_images, errors


def extract_archive(upload_file: UploadFile, temp_dir: str) -> List[str]:
    """
    Extract archive to temp directory.
    Returns list of extracted file paths relative to temp_dir.
    """
    archive_path = os.path.join(temp_dir, "archive")
    
    # Save uploaded file
    with open(archive_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    extracted_files = []
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    filename_lower = upload_file.filename.lower()
    
    if filename_lower.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zf:
            # Check for zip bombs
            total_size = sum(info.file_size for info in zf.infolist())
            if total_size > MAX_ARCHIVE_SIZE_MB * 1024 * 1024 * 10:  # 10x compressed size limit
                raise HTTPException(
                    status_code=400,
                    detail="Archive too large when extracted"
                )
            zf.extractall(extract_dir)
            extracted_files = zf.namelist()
    
    elif filename_lower.endswith(('.tar', '.tar.gz', '.tgz')):
        mode = 'r:gz' if filename_lower.endswith(('.tar.gz', '.tgz')) else 'r'
        with tarfile.open(archive_path, mode) as tf:
            # Security check for path traversal
            for member in tf.getmembers():
                if member.name.startswith('/') or '..' in member.name:
                    raise HTTPException(
                        status_code=400,
                        detail="Archive contains unsafe paths"
                    )
            tf.extractall(extract_dir)
            extracted_files = [m.name for m in tf.getmembers()]
    
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported archive format: {upload_file.filename}"
        )
    
    return extracted_files, extract_dir


async def save_image_to_db(
    file_path: str,
    filename: str,
    user_id: int,
    db: Session,
    caption: str = None,
    instruction: str = None,
    response: str = None,
    content_type: str = None,
    source: str = "bulk_upload",
    task_type: str = "vqa", 
    extra_data: dict = None,
    is_public: bool = False,
) -> dict:
    """
    Save an image to the database and storage.
    Optionally saves caption or instruction.
    Returns the dict with status, image_id, etc .
    """
    #from app.models import Image  # Import here to avoid circular imports

    # Read file data
    with open(file_path, "rb") as f:
        data = f.read()

    if not data:
        return {"status": "error", "error": "Empty file"}

    content_length = len(data)
    sha = _sha256_bytes(data)

    # Exact dedupe
    existing = db.execute(
        select(models.Image).where(
            models.Image.sha256 == sha,
            models.Image.content_length == content_length,
        )
    ).scalar_one_or_none()


    if existing:
        image_id = existing.id
        is_new_image = False
        """
        return {
            "status": "exists_sha",
            "image_id": existing.id,
            "image_path": existing.image_path,
            "sha256": existing.sha256,
        }
        """
    else:
        # Determine extension
        if content_type:
            ext = _ext_from_content_type(content_type)
        else:
            ext = get_file_extension(filename).lstrip(".")
            if not ext:
                ext = "jpg"
    


        # Get filename for new image
        base_fn = _new_image_fn(db)
        new_filename = f"{base_fn}.{ext}"
        path = IMAGES_DIR / new_filename

        # Image conversion and standardization
        img = Image.open(BytesIO(data)).convert("RGB")
        img = _resize_for_saving(img)

        # Write image to disk
        _atomic_write(path, img)
        
        # Compute phash
        phash = _compute_phash(img)


        # Save DB row
        row = models.Image(
            user_id=user_id,
            sha256=sha,
            phash=phash,
            image_url=None,
            image_path=str(path),
            content_length=content_length,
            is_public=is_public,
        )
        db.add(row)

        try:
            db.commit()
            db.refresh(row)
            image_id = row.id
            is_new_image = True
        except IntegrityError:
            db.rollback()
            existing = db.execute(
                select(models.Image).where(models.Image.sha256 == sha)
            ).scalar_one()
            image_id = existing.id
            is_new_image = False
    

    # Save caption if provided
    caption_id = None
    if caption:
        caption_row = models.ImageCaption(
            image_id=image_id,
            user_id=user_id,
            caption=caption,
            source=source,
            extra_data=extra_data or {"original_filename": filename},
        )
        db.add(caption_row)
        db.commit()
        db.refresh(caption_row)
        caption_id = caption_row.id
    

    # Save instruction if provided
    instruction_id = None
    if instruction:
        instruction_row = models.ImageInstruction(
            image_id=image_id,
            user_id=user_id,
            instruction=instruction,
            response=response,
            task_type=task_type,
            source=source,
            extra_data=extra_data or {"original_filename": filename},
        )
        db.add(instruction_row)
        db.commit()
        db.refresh(instruction_row)
        instruction_id = instruction_row.id



    return {
        "status": "inserted" if is_new_image else "exists_sha",
        "image_id": image_id,
        "caption_id": caption_id,
        "instruction_id": instruction_id,
        "original_filename": filename,
    }



# ============== Routes ==============

@router.get("/formats", response_model=List[ArchiveFormat])
async def get_supported_formats():
    """Get documentation on supported upload formats."""
    return [
        ArchiveFormat(
            format_type="multi-image",
            description="Select up to 50 images directly from your device",
            example_structure={
                "allowed_extensions": list(ALLOWED_IMAGE_EXTENSIONS),
                "max_files": MAX_IMAGES_PER_UPLOAD
            }
        ),
        ArchiveFormat(
            format_type="image-archive",
            description="Zip or tar.gz containing only images",
            example_structure={
                "archive.zip": {
                    "image1.jpg": "image file",
                    "image2.png": "image file",
                    "subfolder/": {
                        "image3.jpg": "image file"
                    }
                }
            }
        ),
        ArchiveFormat(
            format_type="image-caption",
            description="Zip with images and a captions.json file",
            example_structure={
                "archive.zip": {
                    "images/": {
                        "photo1.jpg": "image file",
                        "photo2.png": "image file"
                    },
                    "captions.json": [
                        {"filename": "images/photo1.jpg", "caption": "A cat sitting on a couch"},
                        {"filename": "images/photo2.png", "caption": "A sunset over the ocean"}
                    ]
                }
            }
        ),
        ArchiveFormat(
            format_type="image-instruction",
            description="Zip with images and an instructions.json file",
            example_structure={
                "archive.zip": {
                    "images/": {
                        "chart1.png": "image file",
                        "diagram1.jpg": "image file"
                    },
                    "instructions.json": [
                        {
                            "filename": "images/chart1.png",
                            "instruction": "What is the total revenue shown in this chart?",
                            "response": "The total revenue is $1.5 million (optional)"
                        },
                        {
                            "filename": "images/diagram1.jpg",
                            "instruction": "Describe the process shown in this diagram"
                        }
                    ]
                }
            }
        )
    ]


@router.post("/images", response_model=BulkUploadResponse)
async def bulk_upload_images(
    files: List[UploadFile] = File(...),
    is_public: bool = Form(default=False),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Upload multiple images (up to 50) at once.
    """
    if len(files) > MAX_IMAGES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_IMAGES_PER_UPLOAD} images per upload"
        )
    
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    image_ids = []
    errors = []
    
    for file in files:
        if not is_allowed_image(file.filename):
            errors.append(f"Invalid file type: {file.filename}")
            continue
        
        try:

            # Read file content
            content = await file.read()

            # Save to temp file first
            ext = get_file_extension(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Save to database and storage
            result = await save_image_to_db(
                file_path=tmp_path,
                filename=file.filename,
                user_id=current_user.id,
                db=db,
                content_type=file.content_type,
                source="bulk_upload",
                is_public=is_public,
            )

            if result.get("image_id"):
                image_ids.append(result["image_id"])
            else:
                errors.append(f"Failed to save {file.filename}: {result.get('error', 'Unknown Error')}")

            
            # Cleanup temp file
            os.unlink(tmp_path)
            
        except Exception as e:
            errors.append(f"Failed to save {file.filename}: {str(e)}")
            
            # Cleanup temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    return BulkUploadResponse(
        success=len(errors) == 0,
        message=f"Uploaded {len(image_ids)} of {len(files)} images",
        total_files=len(files),
        successful=len(image_ids),
        failed=len(errors),
        image_ids=image_ids,
        errors=errors
    )


@router.post("/archive", response_model=BulkUploadResponse)
async def bulk_upload_archive(
    file: UploadFile = File(...),
    is_public: bool = Form(default=False),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Upload a zip/tar.gz archive containing images.
    All files must be images or folders.
    """
    if not is_allowed_archive(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid archive format. Allowed: {ALLOWED_ARCHIVE_EXTENSIONS}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    size = file.file.tell()
    file.file.seek(0)  # Seek back to start
    
    if size > MAX_ARCHIVE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Archive too large. Maximum size: {MAX_ARCHIVE_SIZE_MB}MB"
        )
    
    image_ids = []
    errors = []

    # Add at the start of each route:
    caption_ids = []
    instruction_ids = []

    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract archive
            extracted_files, extract_dir = extract_archive(file, temp_dir)
            
            # Validate contents
            valid_images, validation_errors = validate_archive_contents(extracted_files)
            errors.extend(validation_errors)
            
            if not valid_images:
                raise HTTPException(
                    status_code=400,
                    detail="No valid images found in archive"
                )
            
            # Process each valid image
            for image_path in valid_images:
                try:
                    full_path = os.path.join(extract_dir, image_path)
                    if not os.path.isfile(full_path):
                        continue
                    
                    # Save to database and storage
                    result = await save_image_to_db(
                        file_path=full_path,
                        filename=os.path.basename(image_path),
                        user_id=current_user.id,
                        db=db,
                        source="bulk_upload",
                        is_public=is_public,
                    )

                    if result.get("image_id"):
                        image_ids.append(result["image_id"])
                    else:
                        errors.append(f"Failed to save {file.filename}: {result.get('error', 'Unknown Error')}")

                    
                except Exception as e:
                    errors.append(f"Failed to save {image_path}: {str(e)}")
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process archive: {str(e)}"
            )
    
    return BulkUploadResponse(
        success=len(errors) == 0,
        message=f"Extracted and uploaded {len(image_ids)} images",
        total_files=len(valid_images) if 'valid_images' in locals() else 0,
        successful=len(image_ids),
        failed=len(errors),
        image_ids=image_ids,
        errors=errors
    )


@router.post("/captions", response_model=BulkUploadResponse)
async def bulk_upload_with_captions(
    file: UploadFile = File(...),
    is_public: bool = Form(default=False),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Upload a zip/tar.gz with images and a captions.json file.
    
    Expected captions.json format:
    [
        {"filename": "image1.jpg", "caption": "Description of image 1"},
        {"filename": "folder/image2.png", "caption": "Description of image 2"}
    ]
    """
    if not is_allowed_archive(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid archive format. Allowed: {ALLOWED_ARCHIVE_EXTENSIONS}"
        )
    
    image_ids = []
    errors = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract archive
            extracted_files, extract_dir = extract_archive(file, temp_dir)
            
            # Validate contents (require captions.json)
            valid_images, validation_errors = validate_archive_contents(
                extracted_files, 
                require_metadata="captions.json"
            )
            errors.extend(validation_errors)
            
            if "Missing required metadata file: captions.json" in errors:
                raise HTTPException(
                    status_code=400,
                    detail="Archive must contain a captions.json file"
                )
            
            # Load captions
            captions_path = os.path.join(extract_dir, "captions.json")
            try:
                with open(captions_path, 'r') as f:
                    captions_data = json.load(f)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid captions.json: {str(e)}"
                )
            
            # Build caption lookup
            caption_map = {}
            for item in captions_data:
                if not isinstance(item, dict) or 'filename' not in item or 'caption' not in item:
                    errors.append(f"Invalid caption entry: {item}")
                    continue
                caption_map[item['filename']] = item['caption']
            
            # Process each valid image
            for image_path in valid_images:
                try:
                    full_path = os.path.join(extract_dir, image_path)
                    if not os.path.isfile(full_path):
                        continue
                    
                    # Get caption (try full path and basename)
                    caption = caption_map.get(image_path) or caption_map.get(os.path.basename(image_path))
                    
                    if not caption:
                        errors.append(f"No caption found for: {image_path}")
                        continue
                    
                    result = await save_image_to_db(
                        file_path=full_path,
                        filename=os.path.basename(image_path),
                        user_id=current_user.id,
                        db=db,
                        caption=caption,
                        source="bulk_upload",
                        extra_data={"original_path": image_path},
                        is_public=is_public,
                    )


                    if result.get("image_id"):
                        image_ids.append(result["image_id"])
                    else:
                        errors.append(f"Failed to save {image_path}: {result.get('error', 'Unknown')}")
                    
                except Exception as e:
                    errors.append(f"Failed to save {image_path}: {str(e)}")
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process archive: {str(e)}"
            )
    
    return BulkUploadResponse(
        success=len(errors) == 0,
        message=f"Uploaded {len(image_ids)} images with captions",
        total_files=len(valid_images) if 'valid_images' in locals() else 0,
        successful=len(image_ids),
        failed=len(errors),
        image_ids=image_ids,
        errors=errors
    )


@router.post("/validate-archive")
async def validate_archive_preview(
    file: UploadFile = File(...),
    format_type: str = Form(default="image-archive"),
    current_user = Depends(get_current_user)
):
    """
    Validate an archive without actually uploading.
    Returns preview of what would be uploaded.
    """
    if not is_allowed_archive(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid archive format. Allowed: {ALLOWED_ARCHIVE_EXTENSIONS}"
        )
    
    require_metadata = None
    if format_type == "image-caption":
        require_metadata = "captions.json"
    elif format_type == "image-instruction":
        require_metadata = "instructions.json"
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            extracted_files, extract_dir = extract_archive(file, temp_dir)
            valid_images, errors = validate_archive_contents(extracted_files, require_metadata)
            
            # If metadata required, parse it for preview
            metadata_preview = None
            if require_metadata:
                metadata_path = os.path.join(extract_dir, require_metadata)
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata_preview = json.load(f)
            
            return {
                "valid": len(errors) == 0,
                "total_files": len(extracted_files),
                "valid_images": len(valid_images),
                "image_list": valid_images[:20],  # First 20 for preview
                "errors": errors,
                "metadata_entries": len(metadata_preview) if metadata_preview else 0,
                "metadata_preview": metadata_preview[:5] if metadata_preview else None
            }
        
        except Exception as e:
            return {
                "valid": False,
                "total_files": 0,
                "valid_images": 0,
                "image_list": [],
                "errors": [str(e)]
            }


@router.post("/instructions", response_model=BulkUploadResponse)
async def bulk_upload_with_instructions(
    file: UploadFile = File(...),
    is_public: bool = Form(default=False),
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Upload a zip/tar.gz with images and an instructions.json file.
    
    Expected instructions.json format:
    [
        {
            "filename": "chart1.png",
            "instruction": "What is shown in this chart?",
            "response": "Optional ground truth response"
        }
    ]
    """
    if not is_allowed_archive(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid archive format. Allowed: {ALLOWED_ARCHIVE_EXTENSIONS}"
        )
    
    image_ids = []
    instruction_ids = []
    errors = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Extract archive
            extracted_files, extract_dir = extract_archive(file, temp_dir)
            
            # Validate contents (require instructions.json)
            valid_images, validation_errors = validate_archive_contents(
                extracted_files,
                require_metadata="instructions.json"
            )
            errors.extend(validation_errors)
            
            if "Missing required metadata file: instructions.json" in errors:
                raise HTTPException(
                    status_code=400,
                    detail="Archive must contain an instructions.json file"
                )
            
            # Load instructions
            instructions_path = os.path.join(extract_dir, "instructions.json")
            try:
                with open(instructions_path, 'r') as f:
                    instructions_data = json.load(f)
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid instructions.json: {str(e)}"
                )
            
            # Build instruction lookup
            instruction_map = {}
            for item in instructions_data:
                if not isinstance(item, dict) or 'filename' not in item or 'instruction' not in item:
                    errors.append(f"Invalid instruction entry: {item}")
                    continue
                instruction_map[item['filename']] = {
                    'instruction': item['instruction'],
                    'response': item.get('response'),
                    'task_type': item.get('task_type', 'vqa'),
                }
            
            if not instruction_map:
                raise HTTPException(
                    status_code=400,
                    detail="No valid instruction entries found in instructions.json"
                )
            
            # Process each valid image
            for image_path in valid_images:
                try:
                    full_path = os.path.join(extract_dir, image_path)
                    if not os.path.isfile(full_path):
                        continue
                    
                    # Get instruction (try full path and basename)
                    instr_data = instruction_map.get(image_path) or instruction_map.get(os.path.basename(image_path))
                    
                    if not instr_data:
                        errors.append(f"No instruction found for: {image_path}")
                        continue
                    
                    result = await save_image_to_db(
                        file_path=full_path,
                        filename=os.path.basename(image_path),
                        user_id=current_user.id,
                        db=db,
                        instruction=instr_data["instruction"],
                        response=instr_data.get("response"),
                        task_type=instr_data.get("task_type", "vqa"),
                        source="bulk_upload",
                        extra_data={"original_path": image_path},
                        is_public=is_public,
                    )
                    
                    if result.get("image_id"):
                        image_ids.append(result["image_id"])
                    if result.get("instruction_id"):
                        instruction_ids.append(result["instruction_id"])
                    
                    if not result.get("image_id"):
                        errors.append(f"Failed to save {image_path}: {result.get('error', 'Unknown')}")
                
                except Exception as e:
                    errors.append(f"Failed to save {image_path}: {str(e)}")
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process archive: {str(e)}"
            )
    
    return BulkUploadResponse(
        success=len(errors) == 0,
        message=f"Uploaded {len(image_ids)} images with {len(instruction_ids)} instructions",
        total_files=len(valid_images) if 'valid_images' in locals() else 0,
        successful=len(image_ids),
        failed=len(errors),
        image_ids=image_ids,
        instruction_ids=instruction_ids,
        errors=errors
    )