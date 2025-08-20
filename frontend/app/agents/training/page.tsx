import { Suspense } from 'react'
import { MLTrainingWorkspace } from '@/components/agents/ml-training-workspace'
import { AgentWorkspaceSkeleton } from '@/components/ui/skeleton'

export default function MLTrainingPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<AgentWorkspaceSkeleton />}>
          <MLTrainingWorkspace />
        </Suspense>
      </div>
    </main>
  )
}

export const metadata = {
  title: 'ML Training - AI Data Science Platform',
  description: 'Train machine learning models with H2O AutoML',
}
