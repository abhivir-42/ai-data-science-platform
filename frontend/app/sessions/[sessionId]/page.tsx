import { Suspense } from 'react'
import { notFound } from 'next/navigation'
import { SessionResultsViewer } from '@/components/session/session-results-viewer'
import { SessionViewerSkeleton } from '@/components/ui/skeleton'

interface SessionPageProps {
  params: {
    sessionId: string
  }
}

export default function SessionPage({ params }: SessionPageProps) {
  const { sessionId } = params

  // Basic validation for session ID format
  if (!sessionId || sessionId.length < 10) {
    notFound()
  }

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        <Suspense fallback={<SessionViewerSkeleton />}>
          <SessionResultsViewer sessionId={sessionId} />
        </Suspense>
      </div>
    </main>
  )
}

export function generateMetadata({ params }: SessionPageProps) {
  return {
    title: `Session ${params.sessionId} - AI Data Science Platform`,
    description: 'View session results with data, code, logs, recommendations, and analysis',
  }
}
