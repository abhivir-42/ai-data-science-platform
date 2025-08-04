"""
File Processing Service

Uses the working Data Loader Agent to provide real data analysis, preview, 
and summary capabilities for uploaded files.
"""

import os
import pandas as pd
from typing import Dict, Any, Optional, List, Union
from loguru import logger
import json
from pathlib import Path

from app.core.config import settings
from app.services.agent_execution import agent_execution_service, AgentExecutionError


class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass


class FileProcessingService:
    """Service for processing uploaded files using AI agents"""
    
    def __init__(self):
        self.execution_service = agent_execution_service
        self.upload_path = Path(settings.UPLOAD_PATH)
    
    def get_file_path(self, file_id: str) -> Optional[str]:
        """Get the full file path for a given file ID"""
        try:
            # Find file by ID prefix
            if not self.upload_path.exists():
                return None
                
            files = [f for f in os.listdir(self.upload_path) if f.startswith(file_id)]
            if not files:
                return None
            
            return str(self.upload_path / files[0])
        except Exception as e:
            logger.error(f"Error finding file {file_id}: {e}")
            return None
    
    def get_original_filename(self, file_id: str) -> Optional[str]:
        """Get the original filename for a given file ID"""
        try:
            files = [f for f in os.listdir(self.upload_path) if f.startswith(file_id)]
            if not files:
                return None
            
            # Remove UUID prefix (36 chars + 1 underscore)
            return files[0][37:]
        except Exception as e:
            logger.error(f"Error getting original filename for {file_id}: {e}")
            return None
    
    async def analyze_file(self, file_id: str) -> Dict[str, Any]:
        """
        Use Data Loader Agent to analyze an uploaded file
        
        Args:
            file_id: ID of the uploaded file
            
        Returns:
            Dictionary containing file analysis results
        """
        file_path = self.get_file_path(file_id)
        if not file_path:
            raise FileProcessingError(f"File not found: {file_id}")
        
        original_filename = self.get_original_filename(file_id)
        
        try:
            # Use Data Loader Agent to analyze the file
            analysis_instruction = f"""
            Analyze the file at path: {file_path}
            
            Please:
            1. Load and examine the data structure
            2. Provide information about columns, data types, and shape
            3. Give a sample of the data (first few rows)
            4. Identify any data quality issues
            5. Summarize the contents and structure
            
            Focus on providing detailed information about the dataset structure and contents.
            """
            
            result = self.execution_service.execute_agent_sync(
                agent_id="data_loader",
                parameters={"user_instructions": analysis_instruction}
            )
            
            if result["status"] == "failed":
                raise FileProcessingError(f"Agent analysis failed: {result.get('message', 'Unknown error')}")
            
            return {
                "file_id": file_id,
                "original_filename": original_filename,
                "file_path": file_path,
                "analysis_status": "completed",
                "agent_result": result.get("result", {}),
                "message": "File analyzed successfully"
            }
            
        except Exception as e:
            logger.error(f"File analysis failed for {file_id}: {e}")
            raise FileProcessingError(f"File analysis failed: {str(e)}")
    
    def extract_data_preview(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract data preview information from agent analysis result
        
        Args:
            analysis_result: Result from analyze_file method
            
        Returns:
            Structured preview data
        """
        try:
            agent_result = analysis_result.get("agent_result", {})
            
            # Try to extract artifacts (actual loaded data)
            artifacts = agent_result.get("artifacts", {})
            ai_message = agent_result.get("ai_message", "")
            
            preview_data = {
                "analysis_summary": ai_message,
                "data_loaded": bool(artifacts),
                "artifacts_info": {},
            }
            
            # If we have actual data artifacts, extract information
            if artifacts:
                preview_data["artifacts_info"] = {
                    "type": str(type(artifacts)),
                    "keys": list(artifacts.keys()) if isinstance(artifacts, dict) else "N/A"
                }
                
                # Try to extract pandas DataFrame information if available
                for key, value in artifacts.items():
                    if isinstance(value, dict) and "data" in str(value).lower():
                        try:
                            # This might be serialized DataFrame data
                            if isinstance(value, dict) and any(k in str(value) for k in ["columns", "index", "data"]):
                                preview_data["dataframe_info"] = {
                                    "artifact_key": key,
                                    "type": "dataframe_like",
                                    "content_preview": str(value)[:500] + "..." if len(str(value)) > 500 else str(value)
                                }
                                break
                        except Exception as e:
                            logger.warning(f"Could not process artifact {key}: {e}")
            
            return preview_data
            
        except Exception as e:
            logger.error(f"Error extracting preview data: {e}")
            return {
                "analysis_summary": analysis_result.get("agent_result", {}).get("ai_message", "Analysis completed"),
                "data_loaded": False,
                "error": str(e)
            }
    
    def get_file_metadata(self, file_id: str) -> Dict[str, Any]:
        """Get metadata for an uploaded file"""
        file_path = self.get_file_path(file_id)
        if not file_path:
            raise FileProcessingError(f"File not found: {file_id}")
        
        original_filename = self.get_original_filename(file_id)
        file_stat = os.stat(file_path)
        
        return {
            "file_id": file_id,
            "original_filename": original_filename,
            "file_path": file_path,
            "file_size": file_stat.st_size,
            "file_type": Path(original_filename).suffix if original_filename else "",
            "upload_time": file_stat.st_ctime,
            "modified_time": file_stat.st_mtime,
        }
    
    async def get_data_summary(self, file_id: str) -> Dict[str, Any]:
        """
        Get comprehensive data summary using agent analysis
        
        Args:
            file_id: ID of the uploaded file
            
        Returns:
            Dictionary containing comprehensive data summary
        """
        try:
            # Get basic file metadata
            metadata = self.get_file_metadata(file_id)
            
            # Perform agent analysis
            analysis_result = await self.analyze_file(file_id)
            
            # Extract structured summary information
            agent_result = analysis_result.get("agent_result", {})
            ai_message = agent_result.get("ai_message", "")
            
            summary = {
                "file_metadata": metadata,
                "analysis_status": analysis_result.get("analysis_status", "unknown"),
                "summary_text": ai_message,
                "agent_analysis": {
                    "has_artifacts": bool(agent_result.get("artifacts")),
                    "tool_calls": agent_result.get("tool_calls", []),
                    "internal_messages": bool(agent_result.get("internal_messages")),
                },
                "processing_info": {
                    "agent_used": "data_loader",
                    "processing_time": "completed",
                    "status": "success"
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary for {file_id}: {e}")
            raise FileProcessingError(f"Data summary failed: {str(e)}")
    
    async def validate_data_quality(self, file_id: str) -> Dict[str, Any]:
        """
        Validate data quality using agent analysis
        
        Args:
            file_id: ID of the uploaded file
            
        Returns:
            Dictionary containing data quality assessment
        """
        try:
            file_path = self.get_file_path(file_id)
            if not file_path:
                raise FileProcessingError(f"File not found: {file_id}")
            
            # Use Data Loader Agent for quality validation
            validation_instruction = f"""
            Examine the file at: {file_path}
            
            Please assess data quality and provide:
            1. Data completeness (missing values, empty cells)
            2. Data consistency (format issues, type mismatches)
            3. Data validity (outliers, invalid values)
            4. Structure quality (headers, column names, organization)
            5. Any recommendations for data cleaning or preprocessing
            
            Focus on identifying potential data quality issues and providing actionable recommendations.
            """
            
            result = self.execution_service.execute_agent_sync(
                agent_id="data_loader",
                parameters={"user_instructions": validation_instruction}
            )
            
            if result["status"] == "failed":
                raise FileProcessingError(f"Data validation failed: {result.get('message', 'Unknown error')}")
            
            return {
                "file_id": file_id,
                "validation_status": "completed",
                "quality_assessment": result.get("result", {}).get("ai_message", ""),
                "agent_analysis": result.get("result", {}),
                "recommendations": [],  # Could be extracted from AI message
                "issues_found": [],     # Could be extracted from AI message
                "overall_quality": "unknown"  # Could be determined from analysis
            }
            
        except Exception as e:
            logger.error(f"Data validation failed for {file_id}: {e}")
            raise FileProcessingError(f"Data validation failed: {str(e)}")


# Global service instance
file_processing_service = FileProcessingService() 