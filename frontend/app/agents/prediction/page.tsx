import { Suspense } from 'react'
import { MLPredictionWorkspace } from '@/components/agents/ml-prediction-workspace'
import { AgentWorkspaceSkeleton } from '@/components/ui/skeleton'

export default function MLPredictionPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<AgentWorkspaceSkeleton />}>
          <MLPredictionWorkspace />
        </Suspense>
      </div>
    </main>
  )
}

export const metadata = {
  title: 'ML Prediction - AI Data Science Platform',
  description: 'Make predictions and analyze models',
}
