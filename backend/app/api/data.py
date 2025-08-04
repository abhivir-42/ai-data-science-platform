"""
Data management API endpoints
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Dict, Any, List
from pydantic import BaseModel
import os
import uuid
from loguru import logger

from app.core.config import settings
from app.services.file_processing import file_processing_service, FileProcessingError

router = APIRouter()


class DataUploadResponse(BaseModel):
    """Response model for data upload"""
    file_id: str
    filename: str
    size: int
    status: str
    message: str


class DataPreviewResponse(BaseModel):
    """Response model for data preview"""
    file_id: str
    filename: str
    preview: Dict[str, Any]
    metadata: Dict[str, Any]


@router.post("/upload")
async def upload_data(file: UploadFile = File(...)) -> DataUploadResponse:
    """Upload a data file"""
    try:
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File extension {file_ext} not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(settings.UPLOAD_PATH, filename)
        
        # Create upload directory if it doesn't exist
        os.makedirs(settings.UPLOAD_PATH, exist_ok=True)
        
        # Save file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            
            # Check file size
            file_size = len(content)
            max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
            if file_size > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB"
                )
            
            buffer.write(content)
        
        logger.info(f"File uploaded: {filename} ({file_size} bytes)")
        
        return DataUploadResponse(
            file_id=file_id,
            filename=file.filename,
            size=file_size,
            status="uploaded",
            message="File uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")


@router.get("/preview/{file_id}")
async def preview_data(file_id: str) -> DataPreviewResponse:
    """Preview uploaded data using AI agent analysis"""
    try:
        # Get file metadata first
        try:
            metadata = file_processing_service.get_file_metadata(file_id)
        except FileProcessingError as e:
            raise HTTPException(status_code=404, detail=str(e))
        
        # Use AI agent to analyze the file
        try:
            analysis_result = await file_processing_service.analyze_file(file_id)
            preview_data = file_processing_service.extract_data_preview(analysis_result)
            
            logger.info(f"Data preview completed for file {file_id}")
            
            return DataPreviewResponse(
                file_id=file_id,
                filename=metadata["original_filename"],
                preview=preview_data,
                metadata={
                    "file_size": metadata["file_size"],
                    "file_type": metadata["file_type"],
                    "upload_time": metadata["upload_time"],
                    "analysis_status": analysis_result.get("analysis_status", "completed")
                }
            )
            
        except FileProcessingError as e:
            logger.error(f"File analysis failed for {file_id}: {e}")
            # Return basic metadata with error info
            return DataPreviewResponse(
                file_id=file_id,
                filename=metadata["original_filename"],
                preview={
                    "analysis_summary": f"Analysis failed: {str(e)}",
                    "data_loaded": False,
                    "error": str(e)
                },
                metadata={
                    "file_size": metadata["file_size"],
                    "file_type": metadata["file_type"],
                    "upload_time": metadata["upload_time"],
                    "analysis_status": "failed"
                }
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data preview failed for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Data preview failed")


@router.post("/validate")
async def validate_data(file_id: str) -> Dict[str, Any]:
    """Validate data quality using AI agent analysis"""
    try:
        # Use file processing service for real validation
        validation_result = await file_processing_service.validate_data_quality(file_id)
        
        logger.info(f"Data validation completed for file {file_id}")
        return validation_result
        
    except FileProcessingError as e:
        logger.error(f"Data validation failed for {file_id}: {e}")
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 500, detail=str(e))
    except Exception as e:
        logger.error(f"Data validation failed for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Data validation failed")


@router.get("/summary/{file_id}")
async def get_data_summary(file_id: str) -> Dict[str, Any]:
    """Get comprehensive data summary using AI agent analysis"""
    try:
        # Use file processing service for real summary
        summary_result = await file_processing_service.get_data_summary(file_id)
        
        logger.info(f"Data summary completed for file {file_id}")
        return summary_result
        
    except FileProcessingError as e:
        logger.error(f"Data summary failed for {file_id}: {e}")
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 500, detail=str(e))
    except Exception as e:
        logger.error(f"Data summary failed for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="Data summary failed")


@router.get("/list")
async def list_uploaded_files() -> List[Dict[str, Any]]:
    """List all uploaded files"""
    try:
        files = []
        if os.path.exists(settings.UPLOAD_PATH):
            for filename in os.listdir(settings.UPLOAD_PATH):
                if "_" in filename:
                    file_id = filename[:36]  # UUID length
                    original_name = filename[37:]
                    file_path = os.path.join(settings.UPLOAD_PATH, filename)
                    
                    files.append({
                        "file_id": file_id,
                        "filename": original_name,
                        "size": os.path.getsize(file_path),
                        "upload_time": os.path.getctime(file_path),
                        "file_type": os.path.splitext(original_name)[1]
                    })
        
        return files
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        raise HTTPException(status_code=500, detail="Failed to list files")


@router.delete("/{file_id}")
async def delete_file(file_id: str) -> Dict[str, str]:
    """Delete an uploaded file"""
    try:
        # Find and delete file
        files = [f for f in os.listdir(settings.UPLOAD_PATH) if f.startswith(file_id)]
        if not files:
            raise HTTPException(status_code=404, detail="File not found")
        
        filename = files[0]
        file_path = os.path.join(settings.UPLOAD_PATH, filename)
        os.remove(file_path)
        
        logger.info(f"File deleted: {filename}")
        return {"message": "File deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File deletion failed for {file_id}: {e}")
        raise HTTPException(status_code=500, detail="File deletion failed") 