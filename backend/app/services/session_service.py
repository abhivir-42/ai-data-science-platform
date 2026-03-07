"""
Centralized Session Service for managing agent sessions with database persistence.

Database-backed service that handles serialization, persistence, and session management
for all agent types.
"""

import json
import pickle
import base64
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from uuid import uuid4

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger
from fastapi import HTTPException

from app.core.database import database_manager
from app.models.session import AgentSession


class AgentSerializationError(Exception):
    """Exception raised when agent serialization/deserialization fails"""
    pass


class SessionNotFoundError(Exception):
    """Exception raised when a session is not found"""
    pass


class SessionService:
    """
    Centralized service for managing agent sessions with database persistence.
    
    This service replaces all individual SessionStore classes and provides:
    - Database-backed session storage
    - Agent instance serialization/deserialization
    - Session timeout management
    - Thread-safe concurrent access
    """
    
    def __init__(self, session_timeout_hours: int = 24):
        """
        Initialize the session service.
        
        Args:
            session_timeout_hours: Hours after which sessions expire (default: 24)
        """
        self.session_timeout_hours = session_timeout_hours
    
    async def create_session(
        self,
        agent_instance: Any,
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Create a new agent session and store it in the database.

        Args:
            agent_instance: The agent instance to store
            agent_type: Type of agent (cleaning, loading, visualization, etc.)
            metadata: Optional metadata to store with the session
            user_id: Optional user ID to associate with the session

        Returns:
            str: The session ID

        Raises:
            AgentSerializationError: If agent serialization fails
        """
        session_id = str(uuid4())
        
        # Initialize enhanced_metadata at the start
        enhanced_metadata = metadata or {}

        try:
            # Try to serialize the agent instance
            try:
                serialized_agent = await self._serialize_agent(agent_instance)
                serialization_successful = True
            except Exception as serialization_error:
                logger.warning(f"Serialization failed for agent type '{agent_type}': {serialization_error}")
                # Fallback: Store minimal metadata for failed serialization
                serialized_agent = {
                    "serialization_failed": True,
                    "error": str(serialization_error),
                    "agent_type": agent_type,
                    "fallback_mode": True,
                    "agent_class": agent_instance.__class__.__name__,
                    "agent_module": agent_instance.__class__.__module__,
                    "fallback_metadata": {
                        "operation": metadata.get("operation") if metadata else None,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                serialization_successful = False

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(hours=self.session_timeout_hours)

            # Enhance metadata with user-friendly information
            enhanced_metadata.update({
                "user_friendly_info": {
                    "agent_display_name": agent_type.replace("_", " ").title(),
                    "created_at_human": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "expires_at_human": expires_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "estimated_size_mb": len(str(serialized_agent).encode('utf-8')) / (1024 * 1024),
                    "data_summary": self._create_data_summary(agent_instance)
                },
                "technical_info": {
                    "serialization_method": "enhanced_extraction",
                    "agent_class": agent_instance.__class__.__name__,
                    "agent_module": agent_instance.__class__.__module__,
                    "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}"
                }
            })

            # Create session record
            session_record = AgentSession(
                session_id=session_id,
                agent_type=agent_type,
                agent_data=serialized_agent,
                session_metadata=enhanced_metadata,
                expires_at=expires_at,
                user_id=user_id
            )

            # Save to database
            async with database_manager.async_session_maker() as db_session:
                db_session.add(session_record)
                await db_session.commit()

            if serialization_successful:
                logger.info(
                    f"Created session {session_id} for agent type '{agent_type}' "
                    f"(expires: {expires_at})"
                )
            else:
                logger.info(
                    f"Created session {session_id} for agent type '{agent_type}' "
                    f"in fallback mode (serialization failed) (expires: {expires_at})"
                )

            return session_id

        except Exception as e:
            logger.error(f"Failed to create session for agent type '{agent_type}': {e}")
            raise AgentSerializationError(f"Session creation failed: {e}")
    
    async def get_session(self, session_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a session by ID, including the deserialized agent instance.
        
        Args:
            session_id: The session ID to retrieve
            user_id: Optional user ID for session ownership validation
            
        Returns:
            Dict containing:
            - agent: The deserialized agent instance
            - created_at: Session creation timestamp (as float)
            - metadata: Session metadata
            - None if session not found or expired
            
        Raises:
            AgentSerializationError: If agent deserialization fails
            ValueError: If user_id is provided but session belongs to different user
        """
        try:
            async with database_manager.async_session_maker() as db_session:
                # Get session from database
                stmt = select(AgentSession).where(AgentSession.session_id == session_id)
                result = await db_session.execute(stmt)
                session_record = result.scalar_one_or_none()
                
                if not session_record:
                    logger.warning(f"Session {session_id} not found")
                    return None
                
                # Validate user ownership if user_id is provided
                if user_id is not None:
                    if session_record.user_id != user_id:
                        logger.warning(f"Session {session_id} access denied: belongs to user {session_record.user_id}, requested by {user_id}")
                        raise ValueError(f"Access denied: Session belongs to another user")
                    logger.debug(f"Session {session_id} ownership validated for user {user_id}")
                
                # Check if session has expired
                if session_record.is_expired():
                    logger.info(f"Session {session_id} has expired, removing")
                    await self.delete_session(session_id)
                    return None
                
                # Update last accessed time
                session_record.last_accessed = datetime.utcnow().replace(tzinfo=None)
                await db_session.commit()

                # Check if this is a fallback session
                agent_data = session_record.agent_data
                if isinstance(agent_data, dict) and agent_data.get("fallback_mode"):
                    logger.warning(f"Session {session_id} is in fallback mode (serialization failed)")
                    # Return a placeholder that indicates fallback mode
                    agent_instance = {
                        "_fallback_mode": True,
                        "_original_error": agent_data.get("error", "Serialization failed"),
                        "_agent_type": session_record.agent_type,
                        "_agent_class": agent_data.get("agent_class"),
                        "_session_id": session_id,
                        "_note": "This session is in fallback mode due to serialization issues. Some functionality may be limited."
                    }
                else:
                    # Deserialize agent instance normally
                    agent_instance = await self._deserialize_agent(session_record.agent_data)

                logger.debug(f"Retrieved session {session_id} (type: {session_record.agent_type})")

                # Handle timezone-aware datetime properly
                created_at_timestamp = session_record.created_at
                if created_at_timestamp.tzinfo is not None:
                    # Convert timezone-aware datetime to UTC timestamp
                    created_at_timestamp = created_at_timestamp.timestamp()
                else:
                    # Handle timezone-naive datetime
                    created_at_timestamp = created_at_timestamp.timestamp()
                
                return {
                    "agent": agent_instance,
                    "created_at": created_at_timestamp,
                    "metadata": session_record.session_metadata or {},
                    "agent_type": session_record.agent_type,
                    "session_id": session_id
                }
                
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            if "deserializ" in str(e).lower():
                raise AgentSerializationError(f"Agent deserialization failed: {e}")
            raise
    
    async def get_session_with_auth(self, session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
        """
        Get a session with authentication, raising HTTPException on errors.
        
        This is a convenience method for FastAPI endpoints that handles
        authentication errors by raising appropriate HTTP exceptions.
        
        Args:
            session_id: The session ID to retrieve
            user_id: The user ID for ownership validation (None for uAgent access)
            
        Returns:
            Dict containing session data
            
        Raises:
            HTTPException: 404 if session not found, 403 if access denied
        """
        try:
            session = await self.get_session(session_id, user_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            return session
        except ValueError as e:
            if "Access denied" in str(e):
                raise HTTPException(status_code=403, detail="Access denied: Session belongs to another user")
            raise
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session by ID.
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            bool: True if session was deleted, False if not found
        """
        try:
            async with database_manager.async_session_maker() as db_session:
                stmt = delete(AgentSession).where(AgentSession.session_id == session_id)
                result = await db_session.execute(stmt)
                await db_session.commit()
                
                deleted = result.rowcount > 0
                if deleted:
                    logger.info(f"Deleted session {session_id}")
                else:
                    logger.warning(f"Session {session_id} not found for deletion")
                
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    async def list_sessions(self, agent_type: Optional[str] = None) -> List[str]:
        """
        List all active (non-expired) session IDs.
        
        Args:
            agent_type: Optional filter by agent type
            
        Returns:
            List of session IDs
        """
        try:
            async with database_manager.async_session_maker() as db_session:
                stmt = select(AgentSession.session_id).where(
                    AgentSession.expires_at > datetime.utcnow().replace(tzinfo=None)
                )
                
                if agent_type:
                    stmt = stmt.where(AgentSession.agent_type == agent_type)
                
                result = await db_session.execute(stmt)
                session_ids = [row[0] for row in result.fetchall()]
                
                logger.debug(f"Found {len(session_ids)} active sessions (type: {agent_type or 'all'})")
                return session_ids
                
        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")
            return []
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Remove all expired sessions from the database.
        
        Returns:
            int: Number of sessions cleaned up
        """
        try:
            async with database_manager.async_session_maker() as db_session:
                stmt = delete(AgentSession).where(
                    AgentSession.expires_at <= datetime.utcnow().replace(tzinfo=None)
                )
                result = await db_session.execute(stmt)
                await db_session.commit()
                
                cleaned_count = result.rowcount
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} expired sessions")
                
                return cleaned_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    async def get_user_friendly_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get user-friendly session information for display in UI

        Returns:
            Dict with human-readable session information
        """
        try:
            async with database_manager.async_session_maker() as db_session:
                stmt = select(AgentSession).where(AgentSession.session_id == session_id)
                result = await db_session.execute(stmt)
                session_record = result.scalar_one_or_none()

                if not session_record:
                    return None

                # Check if expired
                if session_record.is_expired():
                    return {
                        "session_id": session_id,
                        "status": "expired",
                        "message": "This session has expired and is no longer available",
                        "expired_at": session_record.expires_at.isoformat() if session_record.expires_at else None
                    }

                # Extract user-friendly information
                metadata = session_record.session_metadata or {}
                user_info = metadata.get("user_friendly_info", {})
                technical_info = metadata.get("technical_info", {})

                # Get agent data summary
                agent_data = session_record.agent_data
                data_summary = {"has_data": False}

                if isinstance(agent_data, dict):
                    data_summary = agent_data.get("agent_results", {}).get("data_summary", {"has_data": False})

                return {
                    "session_id": session_id,
                    "status": "active",
                    "agent_type": session_record.agent_type,
                    "display_name": user_info.get("agent_display_name", session_record.agent_type.title()),
                    "created_at": user_info.get("created_at_human", session_record.created_at.isoformat() if session_record.created_at else "Unknown"),
                    "expires_at": user_info.get("expires_at_human", session_record.expires_at.isoformat() if session_record.expires_at else "Unknown"),
                    "size_mb": user_info.get("estimated_size_mb", 0),
                    "data_summary": data_summary,
                    "technical_info": technical_info,
                    "last_accessed": session_record.last_accessed.isoformat() if session_record.last_accessed else None
                }

        except Exception as e:
            logger.error(f"Failed to get user-friendly session info for {session_id}: {e}")
            return {
                "session_id": session_id,
                "status": "error",
                "message": f"Could not retrieve session information: {str(e)}"
            }

    async def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.

        Returns:
            Dict with session statistics
        """
        try:
            async with database_manager.async_session_maker() as db_session:
                # Total sessions
                total_stmt = select(AgentSession.session_id)
                total_result = await db_session.execute(total_stmt)
                total_count = len(total_result.fetchall())

                # Active sessions
                active_stmt = select(AgentSession.session_id).where(
                    AgentSession.expires_at > datetime.utcnow().replace(tzinfo=None)
                )
                active_result = await db_session.execute(active_stmt)
                active_count = len(active_result.fetchall())

                # Sessions by type
                type_stmt = select(AgentSession.agent_type).where(
                    AgentSession.expires_at > datetime.utcnow().replace(tzinfo=None)
                )
                type_result = await db_session.execute(type_stmt)
                types = [row[0] for row in type_result.fetchall()]
                type_counts = {}
                for agent_type in types:
                    type_counts[agent_type] = type_counts.get(agent_type, 0) + 1

                # Recently created sessions (last 24 hours)
                recent_stmt = select(AgentSession.session_id).where(
                    AgentSession.created_at > datetime.utcnow().replace(tzinfo=None) - timedelta(hours=24)
                )
                recent_result = await db_session.execute(recent_stmt)
                recent_count = len(recent_result.fetchall())

                # Average session duration (for completed sessions)
                try:
                    avg_duration_stmt = select(
                        (AgentSession.last_accessed - AgentSession.created_at)
                    ).where(
                        AgentSession.last_accessed.isnot(None)
                    )
                    duration_result = await db_session.execute(avg_duration_stmt)
                    durations = []
                    for row in duration_result.fetchall():
                        if row[0] and hasattr(row[0], 'total_seconds'):
                            try:
                                duration_hours = row[0].total_seconds() / 3600
                                durations.append(duration_hours)
                            except (AttributeError, TypeError):
                                continue

                    avg_duration_hours = sum(durations) / len(durations) if durations else 0
                except Exception as duration_error:
                    logger.warning(f"Could not calculate average session duration: {duration_error}")
                    avg_duration_hours = 0

                return {
                    "total_sessions": total_count,
                    "active_sessions": active_count,
                    "expired_sessions": total_count - active_count,
                    "recent_sessions_24h": recent_count,
                    "sessions_by_type": type_counts,
                    "average_session_duration_hours": round(avg_duration_hours, 2),
                    "session_expiration_rate": round((total_count - active_count) / max(total_count, 1) * 100, 1)
                }

        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {
                "total_sessions": 0,
                "active_sessions": 0,
                "expired_sessions": 0,
                "recent_sessions_24h": 0,
                "sessions_by_type": {},
                "average_session_duration_hours": 0,
                "session_expiration_rate": 0
            }
    
    async def _serialize_agent(self, agent_instance: Any) -> Dict[str, Any]:
        """
        Enhanced agent serialization that extracts useful data while avoiding complex objects.
        
        Strategy:
        1. Extract agent results/outputs (the valuable data)
        2. Store agent configuration and metadata
        3. Store reconstruction info for agent recreation
        4. Avoid serializing complex objects like threads, clients, etc.
        
        Args:
            agent_instance: The agent instance to serialize
            
        Returns:
            Dict containing serialized agent data
            
        Raises:
            AgentSerializationError: If serialization fails
        """
        try:
            serialized_data = {
                "serialization_method": "enhanced_extraction",
                "agent_class": agent_instance.__class__.__name__,
                "agent_module": agent_instance.__class__.__module__,
                "timestamp": datetime.utcnow().isoformat(),
                "agent_results": {},
                "agent_config": {},
                "reconstruction_info": {}
            }
            
            # Extract agent results/outputs (the valuable data)
            agent_results = self._extract_agent_results(agent_instance)
            serialized_data["agent_results"] = agent_results
            
            # Extract basic configuration
            agent_config = self._extract_agent_config(agent_instance)
            serialized_data["agent_config"] = agent_config
            
            # Store reconstruction information
            serialized_data["reconstruction_info"] = {
                "requires_reinitialization": True,
                "serializable_attributes": list(agent_config.keys()),
                "extracted_results": list(agent_results.keys())
            }
            
            return serialized_data
            
        except Exception as e:
            raise AgentSerializationError(f"Failed to serialize agent: {e}")
    
    def _extract_agent_results(self, agent_instance: Any) -> Dict[str, Any]:
        """Extract the valuable results/outputs from an agent instance."""
        results = {}
        
        # Common agent result methods to try
        result_methods = [
            # Data cleaning agent
            ('get_data_cleaned', 'cleaned_data'),
            ('get_data_raw', 'raw_data'),
            ('get_data_cleaner_function', 'cleaner_function'),
            ('get_recommended_cleaning_steps', 'cleaning_steps'),
            ('get_workflow_summary', 'workflow_summary'),
            ('get_log_summary', 'log_summary'),
            ('get_response', 'response'),
            
            # Data loader agent
            ('get_artifacts', 'artifacts'),
            ('get_ai_message', 'ai_message'),
            ('get_tool_calls', 'tool_calls'),
            ('get_internal_messages', 'internal_messages'),
            
            # ML Training agent
            ('get_leaderboard', 'leaderboard'),
            ('get_best_model_id', 'best_model_id'),
            ('get_model_path', 'model_path'),
            ('get_h2o_train_function', 'h2o_train_function'),
            ('get_recommended_ml_steps', 'ml_steps'),
            
            # General agent methods
            ('response', 'response_data'),
            ('result', 'result_data'),
            ('output', 'output_data'),
        ]
        
        for method_name, result_key in result_methods:
            try:
                if hasattr(agent_instance, method_name):
                    method = getattr(agent_instance, method_name)
                    if callable(method):
                        try:
                            # Special handling for specific agent methods
                            if method_name == 'get_artifacts':
                                result = method(as_dataframe=True)
                            elif method_name in ['predict_single', 'predict_batch', 'analyze_model']:
                                # ML Prediction agent methods need dummy parameters for serialization
                                if method_name == 'predict_single':
                                    result = method({"dummy": "data"})
                                elif method_name == 'predict_batch':
                                    result = method("dummy.csv")
                                elif method_name == 'analyze_model':
                                    result = method("What is this model?")
                            else:
                                # Try calling method with no args
                                result = method()
                        except TypeError:
                            # Try calling with common args
                            try:
                                result = method(as_dataframe=True)
                            except:
                                continue
                    else:
                        # It's an attribute, not a method
                        result = method
                    
                    # Convert result to JSON-safe format
                    if result is not None:
                        # For DataFrames, check if empty using .empty property
                        if hasattr(result, 'empty') and hasattr(result, 'shape'):
                            if result.empty:
                                continue  # Skip empty DataFrames
                        safe_result = self._make_json_safe(result)
                        results[result_key] = safe_result
                        
            except Exception:
                # Skip problematic methods/attributes
                continue
                
        return results
    
    def _extract_agent_config(self, agent_instance: Any) -> Dict[str, Any]:
        """Extract basic configuration and simple attributes from agent."""
        config = {}
        
        # Common configuration attributes
        config_attrs = [
            'model', 'temperature', 'max_tokens', 'timeout',
            'n_samples', 'log', 'verbose', 'seed', 'api_key',
            'config', 'settings', 'params', 'options'
        ]
        
        for attr_name in config_attrs:
            try:
                if hasattr(agent_instance, attr_name):
                    value = getattr(agent_instance, attr_name)
                    safe_value = self._make_json_safe(value)
                    config[attr_name] = safe_value
            except Exception:
                continue
                
        return config
    
    def _make_json_safe(self, data: Any) -> Any:
        """Convert data to JSON-safe format, handling pandas/numpy types."""
        import pandas as pd
        import numpy as np
        from datetime import datetime, date

        if data is None:
            return None
        # ✅ CRITICAL FIX: Handle NaN values first before type checking
        # Only check for NaN on scalar values, not DataFrames
        elif not isinstance(data, (pd.DataFrame, pd.Series)) and pd.isna(data):
            return None
        elif isinstance(data, (str, int, float, bool)):
            # ✅ Additional safety check for float NaN
            if isinstance(data, float) and np.isnan(data):
                return None
            return data
        elif isinstance(data, dict):
            return {k: self._make_json_safe(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._make_json_safe(item) for item in data]
        elif isinstance(data, (datetime, date)):
            return data.isoformat()
        elif isinstance(data, pd.Timestamp):
            return data.isoformat()
        elif isinstance(data, np.datetime64):
            return pd.Timestamp(data).isoformat()
        elif isinstance(data, pd.Period):
            return str(data)
        elif isinstance(data, pd.Timedelta):
            return str(data)
        elif hasattr(data, 'dtype') and hasattr(data.dtype, 'type'):
            # Check for pandas datetime dtypes
            if np.issubdtype(data.dtype, np.datetime64):
                try:
                    return pd.Timestamp(data).isoformat()
                except:
                    return str(data)
            elif np.issubdtype(data.dtype, np.integer):
                return int(data)
            elif np.issubdtype(data.dtype, np.floating):
                # ✅ CRITICAL FIX: Check for NaN before converting to float
                # Only check for NaN on scalar values, not DataFrames
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    return data
                return float(data) if not np.isnan(data) else None
            elif np.issubdtype(data.dtype, np.bool_):
                return bool(data)
        elif isinstance(data, pd.Series):
            try:
                # Convert to list and make JSON-safe
                series_list = data.tolist()
                return [self._make_json_safe(item) for item in series_list]
            except:
                return list(data)
        elif isinstance(data, pd.Index):
            try:
                # Convert to list and make JSON-safe
                index_list = data.tolist()
                return [self._make_json_safe(item) for item in index_list]
            except:
                return list(data)
        elif isinstance(data, pd.DataFrame):
            try:
                # Convert DataFrame to records format and make JSON-safe
                df_dict = data.to_dict(orient='records')
                # Recursively apply _make_json_safe to each record
                return [self._make_json_safe(record) for record in df_dict]
            except Exception as e:
                print(f"DataFrame serialization failed: {e}")
                return str(data)
        elif hasattr(data, 'isoformat'):  # Other datetime-like objects
            return data.isoformat()
        elif hasattr(data, 'to_dict'):  # Other pandas objects
            try:
                return self._make_json_safe(data.to_dict())
            except Exception as e:
                print(f"Pandas object serialization failed: {e}")
                return str(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, (np.ndarray, np.matrix)):
            try:
                array_list = data.tolist()
                # Recursively apply _make_json_safe to each element
                return [self._make_json_safe(item) for item in array_list]
            except:
                try:
                    flat_list = data.flatten().tolist()
                    return [self._make_json_safe(item) for item in flat_list]
                except:
                    return str(data)
        elif hasattr(data, 'item'):  # Other numpy scalars
            try:
                return data.item()
            except:
                return str(data)
        elif hasattr(data, 'tolist'):  # Other array-like objects
            try:
                return data.tolist()
            except:
                return str(data)
        elif hasattr(data, 'content') and hasattr(data, 'type'):
            # Handle LangChain message objects
            try:
                return {
                    'type': getattr(data, 'type', 'unknown'),
                    'content': str(getattr(data, 'content', '')),
                    'name': getattr(data, 'name', None),
                    'tool_calls': getattr(data, 'tool_calls', None),
                    'usage_metadata': getattr(data, 'usage_metadata', None),
                }
            except:
                return str(data)
        elif hasattr(data, 'asm8') and hasattr(data, 'dtype'):
            # Handle numpy datetime64 scalars that might slip through
            try:
                return pd.Timestamp(data).isoformat()
            except:
                return str(data)
        elif hasattr(data, 'values') and hasattr(data, 'index'):
            # Handle pandas-like objects that might have datetime data
            try:
                if hasattr(data, 'to_dict'):
                    return self._make_json_safe(data.to_dict())
                else:
                    return str(data)
            except:
                return str(data)
        else:
            # Try to identify what type this is for debugging
            data_type = type(data).__name__
            module = type(data).__module__

            # Check if this is a pandas Timestamp that slipped through
            if hasattr(data, 'to_pydatetime'):
                try:
                    return data.to_pydatetime().isoformat()
                except:
                    pass

            # Check if this is a pandas object with datetime conversion
            if hasattr(data, 'to_datetime'):
                try:
                    return str(data.to_datetime())
                except:
                    pass

            # Check for pandas datetime objects with different attributes
            # Only check for datetime attributes on non-DataFrame objects
            if not isinstance(data, (pd.DataFrame, pd.Series)) and (hasattr(data, '_is_naive') or hasattr(data, 'tz_localize')):
                try:
                    # This is likely a pandas datetime-like object
                    return pd.Timestamp(data).isoformat()
                except:
                    pass

            # Check for pandas Period or Interval objects
            if hasattr(data, 'start') and hasattr(data, 'end'):
                try:
                    return f"{data.start} to {data.end}"
                except:
                    pass

            print(f"DEBUG: Unknown data type: {module}.{data_type} = {str(data)[:200]}...")
            # Convert to string as fallback
            return str(data)
    
    async def _deserialize_agent(self, serialized_data: Dict[str, Any]) -> Any:
        """
        Enhanced agent deserialization that creates a functional agent proxy.
        
        Instead of trying to recreate complex agents, we create a proxy object
        that provides access to the extracted results and data.
        
        Args:
            serialized_data: The serialized agent data
            
        Returns:
            Agent proxy with access to extracted results
            
        Raises:
            AgentSerializationError: If deserialization fails
        """
        try:
            method = serialized_data.get("serialization_method")
            
            if method == "enhanced_extraction":
                # Create agent proxy from extracted data
                return self._create_agent_proxy(serialized_data)
                
            elif method == "pickle":
                # Legacy pickle deserialization
                data = serialized_data.get("data")
                decoded_data = base64.b64decode(data.encode('utf-8'))
                agent_instance = pickle.loads(decoded_data)
                return agent_instance
                
            elif method in ["to_dict", "cleaned_dict"]:
                # Legacy JSON deserialization
                return self._create_legacy_proxy(serialized_data)
            
            else:
                raise AgentSerializationError(f"Unknown serialization method: {method}")
                
        except Exception as e:
            raise AgentSerializationError(f"Failed to deserialize agent: {e}")
    
    def _create_agent_proxy(self, serialized_data: Dict[str, Any]) -> Any:
        """Create a functional agent proxy from extracted data."""
        
        class AgentProxy:
            """Proxy object that mimics agent interface with extracted data."""
            
            def __init__(self, agent_data):
                self._agent_class = agent_data.get("agent_class", "Unknown")
                self._agent_module = agent_data.get("agent_module", "Unknown")
                self._results = agent_data.get("agent_results", {})
                self._config = agent_data.get("agent_config", {})
                self._timestamp = agent_data.get("timestamp")
                self._reconstruction_info = agent_data.get("reconstruction_info", {})
                self._agent_data = agent_data  # Store full agent data for fallback access
            
            # Data cleaning agent methods
            def get_data_cleaned(self):
                return self._results.get("cleaned_data")
            
            def get_data_raw(self):
                return self._results.get("raw_data")
            
            def get_data_cleaner_function(self):
                return self._results.get("cleaner_function")
            
            def get_recommended_cleaning_steps(self):
                return self._results.get("cleaning_steps")
            
            def get_workflow_summary(self):
                return self._results.get("workflow_summary")
            
            def get_log_summary(self):
                return self._results.get("log_summary")
            
            def get_response(self):
                return self._results.get("response")
            
            # Feature Engineering agent methods
            def get_data_engineered(self):
                """Get engineered data from feature engineering agent."""
                # Check in response_data first (where Feature Engineering agent stores it)
                response_data = self._results.get("response_data", {})
                engineered_data = response_data.get("data_engineered")
                
                # Fallback to direct key for backward compatibility
                if not engineered_data:
                    engineered_data = self._results.get("data_engineered")
                
                if engineered_data:
                    try:
                        import pandas as pd
                        # Handle different data formats
                        if isinstance(engineered_data, dict):
                            # If it's a dict with column names as keys
                            return pd.DataFrame(engineered_data)
                        elif isinstance(engineered_data, list):
                            # If it's a list of records
                            return pd.DataFrame(engineered_data)
                        else:
                            print(f"[DEBUG] Unexpected engineered data type: {type(engineered_data)}")
                            return None
                    except Exception as e:
                        print(f"[DEBUG] Failed to convert engineered data to DataFrame: {e}")
                        print(f"[DEBUG] Data type: {type(engineered_data)}")
                        if isinstance(engineered_data, (dict, list)) and len(str(engineered_data)) < 500:
                            print(f"[DEBUG] Data preview: {engineered_data}")
                        return None
                return None
            
            def get_feature_engineer_function(self):
                """Get the feature engineering function code."""
                # Check in response_data first
                response_data = self._results.get("response_data", {})
                function_code = response_data.get("feature_engineer_function")
                
                # Fallback to direct key for backward compatibility
                if not function_code:
                    function_code = self._results.get("feature_engineer_function")
                
                return function_code
            
            def get_recommended_feature_engineering_steps(self):
                """Get recommended feature engineering steps."""
                # Check in response_data first
                response_data = self._results.get("response_data", {})
                steps = response_data.get("recommended_steps")
                
                # Fallback to direct key for backward compatibility
                if not steps:
                    steps = self._results.get("recommended_feature_engineering_steps")
                
                return steps
            
            # Data loader agent methods
            def get_artifacts(self, as_dataframe=True):
                artifacts = self._results.get("artifacts")
                if artifacts and as_dataframe:
                    # Convert back to DataFrame-like format if needed
                    if isinstance(artifacts, dict) and "records" in artifacts:
                        try:
                            import pandas as pd
                            return pd.DataFrame(artifacts["records"])
                        except:
                            pass
                return artifacts
            
            def get_ai_message(self):
                return self._results.get("ai_message")
            
            def get_tool_calls(self):
                return self._results.get("tool_calls")
            
            def get_internal_messages(self):
                return self._results.get("internal_messages")
            
            # ML Training agent methods
            def get_leaderboard(self):
                return self._results.get("leaderboard")
            
            def get_best_model_id(self):
                return self._results.get("best_model_id")
            
            def get_model_path(self):
                return self._results.get("model_path")
            
            def get_h2o_train_function(self):
                return self._results.get("h2o_train_function")
            
            def get_recommended_ml_steps(self):
                return self._results.get("ml_steps")
            
            def get_log_summary(self):
                return self._results.get("log_summary")
            
            def get_workflow_summary(self):
                return self._results.get("workflow_summary")
            
            # ML Prediction agent methods
            def predict_single(self, input_data=None):
                return self._results.get("prediction_results")
            
            def predict_batch(self, data_source=None):
                return self._results.get("batch_results")
            
            def analyze_model(self, query=None):
                return self._results.get("model_analysis")
            
            # General properties
            @property
            def response(self):
                # WORKING COMMIT COMPATIBILITY: Map h2o_train_function to training_function
                response_data = self._results.get("response_data") or self._results.get("response")
                if response_data and isinstance(response_data, dict):
                    # Create a copy to avoid modifying original data
                    mapped_response = response_data.copy()
                    # Map h2o_train_function to training_function for backward compatibility
                    if "h2o_train_function" in mapped_response and "training_function" not in mapped_response:
                        mapped_response["training_function"] = mapped_response["h2o_train_function"]
                    return mapped_response
                return response_data
            
            @property
            def result(self):
                return self._results.get("result_data")
            
            @property
            def output(self):
                return self._results.get("output_data")
            
            # Utility methods
            def get_available_methods(self):
                """Return list of available methods with data."""
                available = []
                for key in self._results.keys():
                    if self._results[key] is not None:
                        available.append(key)
                return available
            
            def get_agent_info(self):
                """Return agent metadata."""
                return {
                    "agent_class": self._agent_class,
                    "agent_module": self._agent_module,
                    "timestamp": self._timestamp,
                    "available_results": self.get_available_methods(),
                    "config": self._config
                }
            
            def __repr__(self):
                return f"AgentProxy({self._agent_class}, results: {len(self._results)})"
        
        return AgentProxy(serialized_data)
    
    def _create_legacy_proxy(self, serialized_data: Dict[str, Any]) -> Any:
        """Create proxy for legacy serialization methods."""
        
        class LegacyAgentProxy:
            def __init__(self, data):
                self._data = data.get("data", {})
                self._agent_class = data.get("agent_class", "Unknown")
                self._method = data.get("serialization_method", "unknown")
            
            def get_data(self):
                return self._data
            
            def get_agent_info(self):
                return {
                    "agent_class": self._agent_class,
                    "serialization_method": self._method,
                    "note": "Legacy proxy - limited functionality"
                }
            
            def __repr__(self):
                return f"LegacyAgentProxy({self._agent_class}, method: {self._method})"
        
        return LegacyAgentProxy(serialized_data)
    
    def _create_data_summary(self, agent_instance: Any) -> Dict[str, Any]:
        """Create a user-friendly summary of agent data for display"""
        summary = {
            "has_data": False,
            "data_type": "Unknown",
            "estimated_rows": 0,
            "estimated_columns": 0,
            "data_preview": None
        }

        try:
            # Try to extract meaningful information from the agent
            if hasattr(agent_instance, 'get_data_cleaned'):
                try:
                    data = agent_instance.get_data_cleaned()
                    if data is not None:
                        summary.update(self._analyze_dataframe(data, "cleaned"))
                except:
                    pass

            if not summary["has_data"] and hasattr(agent_instance, 'get_data_raw'):
                try:
                    data = agent_instance.get_data_raw()
                    if data is not None:
                        summary.update(self._analyze_dataframe(data, "raw"))
                except:
                    pass

            # Add agent-specific information
            if hasattr(agent_instance, 'response') and agent_instance.response:
                summary["has_response"] = True
                summary["response_type"] = type(agent_instance.response).__name__

        except Exception as e:
            summary["error"] = f"Could not analyze data: {str(e)}"

        return summary

    def _analyze_dataframe(self, data: Any, data_type: str) -> Dict[str, Any]:
        """Analyze a dataframe-like object and return summary"""
        analysis = {"has_data": False, "data_type": data_type}

        try:
            # Check if it's a pandas DataFrame or similar
            if hasattr(data, 'shape'):
                analysis["has_data"] = True
                analysis["estimated_rows"] = int(data.shape[0])
                analysis["estimated_columns"] = int(data.shape[1])
                analysis["data_shape"] = f"{data.shape[0]} rows × {data.shape[1]} columns"

            # Check if it's a dictionary with records
            elif isinstance(data, dict) and "records" in data:
                records = data["records"]
                if records:
                    analysis["has_data"] = True
                    analysis["estimated_rows"] = len(records)
                    analysis["estimated_columns"] = len(records[0]) if records else 0
                    analysis["data_shape"] = f"{len(records)} records"

            # Try to get column names
            if hasattr(data, 'columns'):
                column_names = list(data.columns[:5])  # First 5 columns
                # Make column names JSON-safe
                analysis["column_names"] = [self._make_json_safe(name) for name in column_names]
                if len(data.columns) > 5:
                    analysis["column_names"].append(f"... and {len(data.columns) - 5} more")

            # Create a small preview
            if hasattr(data, 'head'):
                try:
                    preview = data.head(3).to_dict('records') if hasattr(data, 'to_dict') else None
                    if preview:
                        # Make the preview JSON-safe to prevent serialization issues
                        safe_preview = [self._make_json_safe(record) for record in preview]
                        analysis["data_preview"] = safe_preview
                except:
                    pass

        except Exception as e:
            analysis["analysis_error"] = str(e)

        return analysis

    def _clean_dict_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean a dictionary to make it JSON serializable.

        Removes or converts non-serializable objects.
        """
        cleaned = {}

        for key, value in data.items():
            try:
                if value is None or isinstance(value, (str, int, float, bool, list, dict)):
                    cleaned[key] = value
                elif hasattr(value, 'to_dict'):
                    cleaned[key] = value.to_dict()
                elif hasattr(value, '__dict__'):
                    cleaned[key] = str(value)  # Convert to string representation
                else:
                    cleaned[key] = str(value)
            except Exception:
                # Skip problematic attributes
                continue

        return cleaned


# Global session service instance
session_service = SessionService()
