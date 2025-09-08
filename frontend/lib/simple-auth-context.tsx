import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAppStore } from './store';

interface SimpleUser {
  user_id: string;
  username: string;
  full_name?: string;
}

interface SimpleAuthContextType {
  user: SimpleUser | null;
  login: (username: string, password: string) => Promise<void>;
  register: (userData: any) => Promise<void>;
  logout: () => void;
  isLoading: boolean;
  isAuthenticated: boolean;
}

const SimpleAuthContext = createContext<SimpleAuthContextType | undefined>(undefined);

export function SimpleAuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<SimpleUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for stored session on app load
  useEffect(() => {
    const sessionId = localStorage.getItem('session_id');
    if (sessionId) {
      // Validate session with backend
      validateSession(sessionId);
    } else {
      setIsLoading(false);
    }
  }, []);

  const validateSession = async (sessionId: string) => {
    try {
      const response = await fetch(`/api/auth/me?session_id=${sessionId}`);
      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          // For now, we don't have user details, just set a placeholder
          setUser({
            user_id: data.user_id,
            username: 'User', // We'll get this from backend later
          });
        }
      }
    } catch (error) {
      console.error('Session validation failed:', error);
      localStorage.removeItem('session_id');
    }
    setIsLoading(false);
  };

  const login = async (username: string, password: string) => {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });

    if (response.ok) {
      const data = await response.json();
      if (data.success) {
        // Store session ID
        localStorage.setItem('session_id', data.session_id);

        // Set user state
        setUser({
          user_id: data.user_id,
          username: username,
        });

        // Set current user in session store for session isolation
        useAppStore.getState().setCurrentUser(data.user_id);

        // Set user ID in localStorage for custom storage
        localStorage.setItem('current_user_id', data.user_id);

        // Migrate any legacy sessions to this user
        useAppStore.getState().migrateLegacySessions(data.user_id);
      } else {
        throw new Error(data.message || 'Login failed');
      }
    } else {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Login failed');
    }
  };

  const register = async (userData: any) => {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(userData)
    });

    if (response.ok) {
      const data = await response.json();
      if (data.success) {
        return data;
      } else {
        throw new Error(data.message || 'Registration failed');
      }
    } else {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Registration failed');
    }
  };

  const logout = async () => {
    const sessionId = localStorage.getItem('session_id');
    if (sessionId) {
      try {
        await fetch('/api/auth/logout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId })
        });
      } catch (error) {
        console.error('Logout API call failed:', error);
      }
    }

    // Clear local state and storage
    localStorage.removeItem('session_id');
    setUser(null);

    // Clear current user from session store
    useAppStore.getState().setCurrentUser(null);

    // Clear user ID from localStorage
    localStorage.removeItem('current_user_id');
  };

  const value = {
    user,
    login,
    register,
    logout,
    isLoading,
    isAuthenticated: !!user
  };

  return (
    <SimpleAuthContext.Provider value={value}>
      {children}
    </SimpleAuthContext.Provider>
  );
}

export function useSimpleAuth() {
  const context = useContext(SimpleAuthContext);
  if (context === undefined) {
    throw new Error('useSimpleAuth must be used within a SimpleAuthProvider');
  }
  return context;
}
