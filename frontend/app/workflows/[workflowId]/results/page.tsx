import { Suspense } from 'react'
import { notFound } from 'next/navigation'
import { WorkflowResultsViewer } from '@/components/workflow/workflow-results-viewer'
import { Skeleton } from '@/components/ui/skeleton'

interface WorkflowResultsPageProps {
  params: {
    workflowId: string
  }
}

function WorkflowResultsViewerSkeleton() {
  return (
    <div className="container mx-auto px-4 py-8 space-y-6">
      <div className="space-y-4">
        <Skeleton className="h-8 w-1/3" />
        <Skeleton className="h-4 w-2/3" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
        <Skeleton className="h-32" />
      </div>
      <div className="space-y-4">
        <Skeleton className="h-64" />
        <Skeleton className="h-64" />
        <Skeleton className="h-64" />
      </div>
    </div>
  )
}

export default function WorkflowResultsPage({ params }: WorkflowResultsPageProps) {
  const { workflowId } = params

  // Basic validation for workflow ID format
  if (!workflowId || workflowId.length < 10) {
    notFound()
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<WorkflowResultsViewerSkeleton />}>
          <WorkflowResultsViewer workflowId={workflowId} />
        </Suspense>
      </div>
    </main>
  )
}

export function generateMetadata({ params }: WorkflowResultsPageProps) {
  return {
    title: `Workflow Results - ${params.workflowId}`,
    description: 'Comprehensive workflow execution results and analysis'
  }
}
