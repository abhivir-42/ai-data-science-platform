'use client'

import { Suspense, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import { WorkflowDashboard } from '@/components/dashboard/workflow-dashboard'
import { DashboardSkeleton } from '@/components/ui/skeleton'
import { useSimpleAuth } from '@/lib/simple-auth-context'

export default function HomePage() {
  const { isAuthenticated, isLoading } = useSimpleAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push('/auth')
    }
  }, [isAuthenticated, isLoading, router])

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <main className="min-h-screen bg-background">
        <DashboardSkeleton />
      </main>
    )
  }

  // Don't render anything if not authenticated (will redirect)
  if (!isAuthenticated) {
    return null
  }

  return (
    <main className="min-h-screen bg-background">
      <Suspense fallback={<DashboardSkeleton />}>
        <WorkflowDashboard />
      </Suspense>
    </main>
  )
}
