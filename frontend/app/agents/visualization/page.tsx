import { Suspense } from 'react'
import { DataVisualizationWorkspace } from '@/components/agents/data-visualization-workspace'
import { AgentWorkspaceSkeleton } from '@/components/ui/skeleton'

export default function DataVisualizationPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<AgentWorkspaceSkeleton />}>
          <DataVisualizationWorkspace />
        </Suspense>
      </div>
    </main>
  )
}

export const metadata = {
  title: 'Data Visualization - AI Data Science Platform',
  description: 'Create interactive charts and visualizations with Plotly',
}
