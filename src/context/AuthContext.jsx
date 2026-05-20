import { createContext, useContext, useEffect, useState, useRef } from 'react';

const AuthContext = createContext();

/* Storage keys for session management */
const SESSION_TOKEN_KEY = 'auth_token';
const SESSION_EXPIRY_KEY = 'auth_token_expiry';

/**
 * Decode JWT payload without verification (for client-side expiry checking only)
 * Do NOT use for security-critical decisions
 */
function decodeJwtPayload(token) {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) return null;
    
    const decoded = JSON.parse(atob(parts[1]));
    return decoded;
  } catch {
    return null;
  }
}

/**
 * Get expiration time in milliseconds from JWT token
 */
function getTokenExpiryMs(token) {
  const payload = decodeJwtPayload(token);
  if (!payload || !payload.exp) return null;
  // exp is in seconds, convert to milliseconds
  return payload.exp * 1000;
}

/**
 * Check if token is expired
 */
function isTokenExpired(token) {
  const expiryMs = getTokenExpiryMs(token);
  if (!expiryMs) return false;
  return Date.now() >= expiryMs;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const expiryCheckInterval = useRef(null);

  /**
   * Clear all session data and cleanup intervals
   */
  const clearSession = () => {
    localStorage.removeItem(SESSION_TOKEN_KEY);
    localStorage.removeItem(SESSION_EXPIRY_KEY);
    setUser(null);
    if (expiryCheckInterval.current) {
      clearInterval(expiryCheckInterval.current);
    }
  };

  /**
   * Setup periodic token expiry check (every minute)
   */
  const setupExpiryCheck = (token) => {
    // Clear any existing interval
    if (expiryCheckInterval.current) {
      clearInterval(expiryCheckInterval.current);
    }

    // Check every minute if token is expiring soon
    expiryCheckInterval.current = setInterval(() => {
      const token = localStorage.getItem(SESSION_TOKEN_KEY);
      if (!token) {
        clearSession();
        return;
      }

      if (isTokenExpired(token)) {
        console.warn('Session expired due to inactivity');
        clearSession();
      }
    }, 60000); // Check every 60 seconds
  };

  // Check if user is logged in on app start
  useEffect(() => {
    const token = localStorage.getItem(SESSION_TOKEN_KEY);
    
    if (token) {
      // Check if token is already expired
      if (isTokenExpired(token)) {
        clearSession();
        setLoading(false);
        return;
      }

      // Verify token with backend
      fetch('/auth/me', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
        .then((res) => {
          if (res.status === 401) {
            // Token rejected by backend
            clearSession();
            return null;
          }
          return res.json();
        })
        .then((data) => {
          if (data?.id) {
            setUser(data);
            setupExpiryCheck(token);
          } else {
            clearSession();
          }
        })
        .catch(() => {
          clearSession();
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }

    // Cleanup on unmount
    return () => {
      if (expiryCheckInterval.current) {
        clearInterval(expiryCheckInterval.current);
      }
    };
  }, []);

  const login = async (email, password) => {
    const response = await fetch('/auth/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
      body: new URLSearchParams({
        username: email,
        password: password,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Login failed');
    }

    const data = await response.json();
    localStorage.setItem(SESSION_TOKEN_KEY, data.access_token);
    
    // Store expiration time for reference
    const expiryMs = getTokenExpiryMs(data.access_token);
    if (expiryMs) {
      localStorage.setItem(SESSION_EXPIRY_KEY, expiryMs.toString());
    }

    // Setup expiry check
    setupExpiryCheck(data.access_token);
    
    setUser(await getCurrentUser());
    return data;
  };

  const register = async (email, displayName, password) => {
    const response = await fetch('/auth/register', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email,
        display_name: displayName,
        password,
      }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Registration failed');
    }

    const data = await response.json();
    return data;
  };

  const logout = () => {
    clearSession();
  };

  const getCurrentUser = async () => {
    const token = localStorage.getItem(SESSION_TOKEN_KEY);
    if (!token) return null;

    // Check if token is expired before making request
    if (isTokenExpired(token)) {
      clearSession();
      return null;
    }

    const response = await fetch('/auth/me', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    if (response.status === 401) {
      // Token was rejected by backend (might be expired/invalid)
      clearSession();
      return null;
    }

    if (!response.ok) return null;
    return await response.json();
  };

  const value = {
    user,
    loading,
    login,
    register,
    logout,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}