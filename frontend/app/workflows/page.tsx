import { Suspense } from 'react'
import { WorkflowBuilder } from '@/components/workflow/workflow-builder'
import { WorkflowBuilderSkeleton } from '@/components/ui/skeleton'

export default function WorkflowsPage() {
  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<WorkflowBuilderSkeleton />}>
          <WorkflowBuilder />
        </Suspense>
      </div>
    </main>
  )
}

export const metadata = {
  title: 'Workflow Builder - AI Data Science Platform',
  description: 'Create and manage multi-step data science workflows',
}
