"""
Simple session manager for prototype.
Uses in-memory storage for user sessions - easily replaceable with Redis later.
"""
import uuid
from datetime import datetime, timedelta
from typing import Optional, Dict

class SimpleSessionManager:
    """Simple session management for prototype - uses in-memory storage"""

    def __init__(self):
        # In-memory session storage (use Redis in production)
        self.sessions: Dict[str, Dict] = {}
        # Session timeout (24 hours for prototype)
        self.session_timeout_hours = 24

    def create_session(self, user_id: str) -> str:
        """Create a new session for user"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=self.session_timeout_hours)
        }
        return session_id

    def get_user_from_session(self, session_id: str) -> Optional[str]:
        """Get user_id from session, returns None if invalid/expired"""
        session = self.sessions.get(session_id)
        if not session:
            return None

        # Check if session expired
        if datetime.utcnow() > session["expires_at"]:
            # Remove expired session
            self.destroy_session(session_id)
            return None

        return session["user_id"]

    def destroy_session(self, session_id: str):
        """Destroy a session"""
        self.sessions.pop(session_id, None)

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session information for debugging"""
        return self.sessions.get(session_id)

    def cleanup_expired_sessions(self):
        """Clean up expired sessions (call this periodically)"""
        expired_sessions = []
        now = datetime.utcnow()

        for session_id, session_data in self.sessions.items():
            if now > session_data["expires_at"]:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            self.destroy_session(session_id)

        return len(expired_sessions)
