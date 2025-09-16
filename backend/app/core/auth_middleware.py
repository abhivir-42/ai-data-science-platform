"""
Authentication middleware for extracting and validating user sessions.

This middleware handles both cookie-based and header-based authentication
and provides user_id extraction for all protected endpoints.
"""

from typing import Optional, Union
from fastapi import HTTPException, Request, Cookie, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.services.simple_session_manager import SimpleSessionManager

class AuthMiddleware:
    """Authentication middleware for user session validation"""
    
    def __init__(self, session_manager: SimpleSessionManager):
        self.session_manager = session_manager
        self.security = HTTPBearer(auto_error=False)  # Optional bearer token
    
    def get_user_from_request(
        self, 
        request: Request,
        session_id: Optional[str] = None,
        authorization: Optional[str] = None
    ) -> Optional[str]:
        """
        Extract user_id from various authentication methods.
        
        Priority:
        1. session_id parameter (direct)
        2. session_id cookie
        3. Authorization header
        4. session_id query parameter
        
        Returns user_id or None if not authenticated
        """
        
        # Method 1: Direct session_id parameter
        if session_id:
            user_id = self.session_manager.get_user_from_session(session_id)
            if user_id:
                return user_id
        
        # Method 2: Session cookie
        session_cookie = request.cookies.get("session_id")
        if session_cookie:
            user_id = self.session_manager.get_user_from_session(session_cookie)
            if user_id:
                return user_id
        
        # Method 3: Authorization header (Bearer token = session_id)
        if authorization:
            # Remove 'Bearer ' prefix if present
            token = authorization.replace('Bearer ', '') if authorization.startswith('Bearer ') else authorization
            user_id = self.session_manager.get_user_from_session(token)
            if user_id:
                return user_id
        
        # Method 4: Query parameter session_id
        query_session = request.query_params.get("session_id")
        if query_session:
            user_id = self.session_manager.get_user_from_session(query_session)
            if user_id:
                return user_id
        
        return None
    
    def require_authentication(
        self,
        request: Request,
        session_id: Optional[str] = None,
        authorization: Optional[str] = None
    ) -> str:
        """
        Require authentication and return user_id.
        Raises HTTPException if not authenticated.
        """
        user_id = self.get_user_from_request(request, session_id, authorization)
        
        if not user_id:
            raise HTTPException(
                status_code=401, 
                detail="Authentication required. Please provide a valid session_id."
            )
        
        return user_id
    
    def optional_authentication(
        self,
        request: Request,
        session_id: Optional[str] = None,
        authorization: Optional[str] = None
    ) -> Optional[str]:
        """
        Optional authentication - returns user_id if authenticated, None otherwise.
        Does not raise exceptions.
        """
        return self.get_user_from_request(request, session_id, authorization)

# Create a shared session manager instance
from app.services.simple_session_manager import SimpleSessionManager

# Global shared session manager - this will be used by both auth_middleware and simple_auth
_shared_session_manager = SimpleSessionManager()

# Global instance with shared session manager
auth_middleware = AuthMiddleware(_shared_session_manager)

# FastAPI Dependencies
def get_current_user_id(
    request: Request,
    session_id: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None)
) -> str:
    """FastAPI dependency for required authentication"""
    return auth_middleware.require_authentication(request, session_id, authorization)

def get_optional_user_id(
    request: Request,
    session_id: Optional[str] = Cookie(None),
    authorization: Optional[str] = Header(None)
) -> Optional[str]:
    """FastAPI dependency for optional authentication"""
    return auth_middleware.optional_authentication(request, session_id, authorization)

# Utility function for manual extraction (for uAgent endpoints)
def extract_user_id_from_request(request) -> Optional[str]:
    """
    Utility function to extract user_id from any request.
    Used in uAgent endpoints that need manual authentication checking.
    
    For uAgents, the request might be a different object, so we handle both cases.
    """
    # Handle FastAPI Request objects
    if hasattr(request, 'headers') and hasattr(request, 'cookies'):
        return auth_middleware.get_user_from_request(request)
    
    # Handle uAgent contexts - return None for now (uAgents need different auth approach)
    # TODO: Implement uAgent-specific authentication
    # For now, uAgents work without user authentication since they're internal services
    return None

def require_user_id_from_request(request: Request) -> str:
    """
    Utility function to require authentication from any request.
    Raises HTTPException if not authenticated.
    """
    return auth_middleware.require_authentication(request)
