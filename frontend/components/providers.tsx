'use client'

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StoreProvider } from '@/providers/store-provider'
import { SimpleAuthProvider } from '@/lib/simple-auth-context'

interface ProvidersProps {
  children: React.ReactNode
}

export function Providers({ children }: ProvidersProps) {
  return (
    <SimpleAuthProvider>
      <StoreProvider>
        <QueryClientProvider client={new QueryClient()}>
          {children}
        </QueryClientProvider>
      </StoreProvider>
    </SimpleAuthProvider>
  )
}
