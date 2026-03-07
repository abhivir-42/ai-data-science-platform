"""
Database models for agent sessions and workflow execution.
"""

from sqlalchemy import Column, String, JSON, DateTime, Integer, Enum
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from enum import Enum as PyEnum

Base = declarative_base()


class WorkflowStatus(PyEnum):
    """Workflow execution status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentSession(Base):
    """Database model for agent sessions"""
    __tablename__ = "agent_sessions"

    session_id = Column(String, primary_key=True)
    agent_type = Column(String, nullable=False)
    agent_data = Column(JSON, nullable=True)
    session_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    user_id = Column(String, nullable=True)

    def is_expired(self) -> bool:
        """Check if this session has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at


class WorkflowExecution(Base):
    """Database model for workflow executions"""
    __tablename__ = "workflow_executions"
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    status = Column(Enum(WorkflowStatus), default=WorkflowStatus.PENDING)
    steps = Column(JSON, nullable=False)
    results = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    current_step_index = Column(Integer, default=0)
    total_execution_time = Column(String, nullable=True)  # Store as string for compatibility
    user_id = Column(String, nullable=True)
    session_metadata = Column(JSON, nullable=True)  # Renamed to avoid SQLAlchemy reserved word
