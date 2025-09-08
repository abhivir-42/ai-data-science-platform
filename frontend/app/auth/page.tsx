'use client'

import React, { useState } from 'react';
import { SimpleLoginForm } from '@/components/auth/simple-login-form';
import { SimpleRegisterForm } from '@/components/auth/simple-register-form';

export default function AuthPage() {
  const [mode, setMode] = useState<'login' | 'register'>('login');

  const handleAuthSuccess = () => {
    // Redirect to main app or dashboard
    window.location.href = '/';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {mode === 'login' ? (
          <SimpleLoginForm
            onSuccess={handleAuthSuccess}
            onSwitchToRegister={() => setMode('register')}
          />
        ) : (
          <SimpleRegisterForm
            onSuccess={() => {
              // After successful registration, switch to login
              setMode('login');
            }}
            onSwitchToLogin={() => setMode('login')}
          />
        )}
      </div>
    </div>
  );
}
