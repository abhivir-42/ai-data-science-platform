import { Suspense } from 'react'
import { DataCleaningWorkspace } from '@/components/agents/data-cleaning-workspace'
import { AgentWorkspaceSkeleton } from '@/components/ui/skeleton'

export default function DataCleaningPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<AgentWorkspaceSkeleton />}>
          <DataCleaningWorkspace />
        </Suspense>
      </div>
    </main>
  )
}

export const metadata = {
  title: 'Data Cleaning - AI Data Science Platform',
  description: 'Clean and preprocess your data with AI recommendations',
}
