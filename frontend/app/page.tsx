import { Suspense } from 'react'
import { WorkflowDashboard } from '@/components/dashboard/workflow-dashboard'
import { DashboardSkeleton } from '@/components/ui/skeleton'

export default function HomePage() {
  return (
    <main className="min-h-screen bg-background">
      <Suspense fallback={<DashboardSkeleton />}>
        <WorkflowDashboard />
      </Suspense>
    </main>
  )
}
