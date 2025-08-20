import { Suspense } from 'react'
import { DataLoaderWorkspace } from '@/components/agents/data-loader-workspace'
import { AgentWorkspaceSkeleton } from '@/components/ui/skeleton'

export default function DataLoaderPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<AgentWorkspaceSkeleton />}>
          <DataLoaderWorkspace />
        </Suspense>
      </div>
    </main>
  )
}

export const metadata = {
  title: 'Data Loader - AI Data Science Platform',
  description: 'Upload and load datasets (CSV, Excel, JSON, PDF, Parquet) for analysis',
}
