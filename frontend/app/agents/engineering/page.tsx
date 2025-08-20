import { Suspense } from 'react'
import { FeatureEngineeringWorkspace } from '@/components/agents/feature-engineering-workspace'
import { AgentWorkspaceSkeleton } from '@/components/ui/skeleton'

export default function FeatureEngineeringPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<AgentWorkspaceSkeleton />}>
          <FeatureEngineeringWorkspace />
        </Suspense>
      </div>
    </main>
  )
}

export const metadata = {
  title: 'Feature Engineering - AI Data Science Platform',
  description: 'Engineer and transform features for machine learning',
}
