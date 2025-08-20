'use client'

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { ArrowLeft, RefreshCw, Download, ExternalLink, Database, Code, FileText, Lightbulb, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { DataFrameViewer } from '@/components/core/dataframe-viewer'
import { CodeViewer } from '@/components/core/code-viewer'
import { PlotlyChart } from '@/components/core/plotly-chart'
import { ProgressIndicator } from '@/components/core/progress-indicator'
import { Skeleton } from '@/components/ui/skeleton'
import { useSessionsStore } from '@/lib/store'
import { getAgentClient, type AgentType } from '@/lib/uagent-client'
import { cn, formatRelativeTime, formatDuration } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'

interface SessionResultsViewerProps {
  sessionId: string
}

const agentConfig = {
  loading: { title: 'Data Loader', color: 'blue' },
  cleaning: { title: 'Data Cleaning', color: 'green' },
  visualization: { title: 'Data Visualization', color: 'purple' },
  engineering: { title: 'Feature Engineering', color: 'orange' },
  training: { title: 'ML Training', color: 'indigo' },
  prediction: { title: 'ML Prediction', color: 'red' },
} as const

export function SessionResultsViewer({ sessionId }: SessionResultsViewerProps) {
  const [activeTab, setActiveTab] = useState('data')
  const { getSession } = useSessionsStore()
  const { toast } = useToast()
  
  const session = getSession(sessionId)

  // Get the appropriate client for this session
  const agentClient = session ? getAgentClient(session.agentType) : null

  // Queries for different session results
  const dataQuery = useQuery({
    queryKey: ['session-data', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      
      try {
        // Try different data endpoints based on agent type
        if (session?.agentType === 'cleaning') {
          return await agentClient.getCleanedData(sessionId)
        } else {
          return await agentClient.getSessionData(sessionId)
        }
      } catch (error) {
        console.warn('Data not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!session,
    retry: 1,
    staleTime: 30000,
  })

  const codeQuery = useQuery({
    queryKey: ['session-code', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      
      try {
        return await agentClient.getSessionCode(sessionId)
      } catch (error) {
        console.warn('Code not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!session && ['cleaning', 'visualization', 'engineering', 'training'].includes(session.agentType),
    retry: 1,
    staleTime: 30000,
  })

  const chartQuery = useQuery({
    queryKey: ['session-chart', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      return await agentClient.getSessionChart(sessionId)
    },
    enabled: !!agentClient && !!session && session.agentType === 'visualization',
    retry: 1,
    staleTime: 30000,
  })

  const leaderboardQuery = useQuery({
    queryKey: ['session-leaderboard', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      return await agentClient.getSessionLeaderboard(sessionId)
    },
    enabled: !!agentClient && !!session && session.agentType === 'training',
    retry: 1,
    staleTime: 30000,
  })

  const logsQuery = useQuery({
    queryKey: ['session-logs', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      
      try {
        return await agentClient.getSessionLogs(sessionId)
      } catch (error) {
        console.warn('Logs not available:', error)
        return { logs: [], messages: [] }
      }
    },
    enabled: !!agentClient && !!session,
    retry: 1,
    staleTime: 30000,
  })

  const recommendationsQuery = useQuery({
    queryKey: ['session-recommendations', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      
      try {
        return await agentClient.getSessionRecommendations(sessionId)
      } catch (error) {
        console.warn('Recommendations not available:', error)
        return { recommendations: [] }
      }
    },
    enabled: !!agentClient && !!session,
    retry: 1,
    staleTime: 30000,
  })

  const analysisQuery = useQuery({
    queryKey: ['session-analysis', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      
      try {
        return await agentClient.getSessionAnalysis(sessionId)
      } catch (error) {
        console.warn('Analysis not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!session && session.agentType === 'prediction',
    retry: 1,
    staleTime: 30000,
  })

  // Available tabs based on session type and data availability
  const availableTabs = [
    {
      id: 'data',
      label: 'Data',
      icon: Database,
      available: !!dataQuery.data?.data,
      description: 'View processed dataset'
    },
    {
      id: 'code',
      label: 'Code',
      icon: Code,
      available: !!(codeQuery.data?.code || codeQuery.data?.generated_code),
      description: 'Generated Python functions'
    },
    {
      id: 'logs',
      label: 'Logs',
      icon: FileText,
      available: true,
      description: 'Execution logs and messages'
    },
    {
      id: 'recommendations',
      label: 'Recommendations',
      icon: Lightbulb,
      available: !!recommendationsQuery.data && (recommendationsQuery.data.recommendations?.length || 0) > 0,
      description: 'AI suggestions and next steps'
    },
    {
      id: 'analysis',
      label: 'Analysis',
      icon: BarChart3,
      available: !!chartQuery.data?.figure || !!leaderboardQuery.data?.leaderboard || !!analysisQuery.data,
      description: 'Charts, models, and analysis results'
    },
  ]

  const handleRefresh = () => {
    dataQuery.refetch()
    codeQuery.refetch()
    chartQuery.refetch()
    leaderboardQuery.refetch()
    logsQuery.refetch()
    recommendationsQuery.refetch()
    analysisQuery.refetch()
    
    toast({
      title: "Refreshed",
      description: "Session results have been refreshed.",
      variant: "default",
    })
  }

  if (!session) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Card>
          <CardContent className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="text-4xl mb-4">❓</div>
              <h3 className="text-lg font-medium mb-2">Session Not Found</h3>
              <p className="text-muted-foreground mb-4">
                The session with ID {sessionId} was not found in your local storage.
              </p>
              <Link href="/">
                <Button>Return to Dashboard</Button>
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  const agentInfo = agentConfig[session.agentType]

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b bg-white">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <Button variant="ghost" size="sm">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Dashboard
                </Button>
              </Link>
              
              <div className="space-y-1">
                <div className="flex items-center space-x-3">
                  <h1 className="text-2xl font-bold">Session Results</h1>
                  <Badge 
                    variant={
                      session.status === 'completed' ? 'secondary' : 
                      session.status === 'failed' ? 'destructive' : 'default'
                    }
                  >
                    {session.status}
                  </Badge>
                </div>
                
                <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                  <span>{agentInfo.title}</span>
                  <span>•</span>
                  <span>Session {sessionId.slice(0, 8)}...</span>
                  <span>•</span>
                  <span>{formatRelativeTime(session.createdAt)}</span>
                  {session.executionTimeSeconds && (
                    <>
                      <span>•</span>
                      <span>{formatDuration(session.executionTimeSeconds)}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm" onClick={handleRefresh}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Refresh
              </Button>
              
              <Link href={`/agents/${session.agentType}`}>
                <Button variant="outline" size="sm">
                  <ExternalLink className="mr-2 h-4 w-4" />
                  Open Workspace
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {/* Session Status */}
        {session.status === 'running' && (
          <div className="mb-6">
            <ProgressIndicator
              status="running"
              message="Processing your request..."
              startTime={session.createdAt}
              variant="detailed"
            />
          </div>
        )}

        {/* Results Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid w-full grid-cols-5 h-auto p-1">
            {availableTabs.map(tab => {
              const Icon = tab.icon
              return (
                <TabsTrigger
                  key={tab.id}
                  value={tab.id}
                  disabled={!tab.available}
                  className={cn(
                    'flex flex-col items-center space-y-2 p-4 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground',
                    !tab.available && 'opacity-50'
                  )}
                  title={tab.available ? tab.description : 'Not available for this session'}
                >
                  <Icon className="h-5 w-5" />
                  <span className="text-xs">{tab.label}</span>
                </TabsTrigger>
              )
            })}
          </TabsList>

          {/* Data Tab */}
          <TabsContent value="data" className="space-y-4">
            {dataQuery.isLoading ? (
              <Skeleton className="h-96 w-full" />
            ) : dataQuery.data?.data ? (
              <DataFrameViewer
                data={dataQuery.data.data}
                title="Dataset"
                description="Processed dataset from the agent"
                downloadUrls={{
                  csv: '#', // TODO: Implement actual download URLs
                  json: '#',
                }}
              />
            ) : (
              <Card>
                <CardContent className="flex items-center justify-center h-64">
                  <div className="text-center text-muted-foreground">
                    <Database className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No data available for this session</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Code Tab */}
          <TabsContent value="code" className="space-y-4">
            {codeQuery.isLoading ? (
              <Skeleton className="h-96 w-full" />
            ) : (codeQuery.data?.code || codeQuery.data?.generated_code) ? (
              <CodeViewer
                code={codeQuery.data.code || codeQuery.data.generated_code || ''}
                title="Generated Code"
                description="Python functions generated by the agent"
                downloadFileName={`${session.agentType}_${sessionId.slice(0, 8)}`}
              />
            ) : (
              <Card>
                <CardContent className="flex items-center justify-center h-64">
                  <div className="text-center text-muted-foreground">
                    <Code className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No code available for this session</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Logs Tab */}
          <TabsContent value="logs" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Execution Logs</CardTitle>
                <CardDescription>
                  Detailed execution logs and internal messages
                </CardDescription>
              </CardHeader>
              <CardContent>
                {logsQuery.isLoading ? (
                  <div className="space-y-2">
                    {[...Array(5)].map((_, i) => (
                      <Skeleton key={i} className="h-4 w-full" />
                    ))}
                  </div>
                ) : (
                  <div className="space-y-2 max-h-96 overflow-auto">
                    {logsQuery.data?.logs?.map((log, index) => (
                      <div key={index} className="text-sm font-mono bg-muted/50 p-2 rounded">
                        {log}
                      </div>
                    )) || (
                      <p className="text-muted-foreground text-center py-8">
                        No logs available
                      </p>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Recommendations Tab */}
          <TabsContent value="recommendations" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>AI Recommendations</CardTitle>
                <CardDescription>
                  Suggested next steps and optimizations
                </CardDescription>
              </CardHeader>
              <CardContent>
                {recommendationsQuery.isLoading ? (
                  <div className="space-y-2">
                    {[...Array(3)].map((_, i) => (
                      <Skeleton key={i} className="h-12 w-full" />
                    ))}
                  </div>
                ) : recommendationsQuery.data?.recommendations?.length ? (
                  <div className="space-y-3">
                    {recommendationsQuery.data.recommendations.map((rec, index) => (
                      <div key={index} className="flex items-start space-x-3 p-3 rounded-lg bg-muted/50">
                        <Lightbulb className="h-5 w-5 text-yellow-500 mt-0.5" />
                        <div className="text-sm whitespace-pre-wrap font-mono bg-slate-50 p-3 rounded border">
                          {rec}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-muted-foreground text-center py-8">
                    No recommendations available
                  </p>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Analysis Tab */}
          <TabsContent value="analysis" className="space-y-4">
            {session.agentType === 'visualization' && chartQuery.data?.figure ? (
              <PlotlyChart
                figure={chartQuery.data.figure as any}
                title="Generated Visualization"
                description="Interactive chart created by the visualization agent"
              />
            ) : null}
            
            {session.agentType === 'training' && leaderboardQuery.data?.leaderboard && (
              <Card>
                <CardHeader>
                  <CardTitle>Model Leaderboard</CardTitle>
                  <CardDescription>
                    H2O AutoML model performance comparison
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <DataFrameViewer
                    data={{
                      records: leaderboardQuery.data.leaderboard,
                      columns: Object.keys(leaderboardQuery.data.leaderboard[0] || {})
                    }}
                    title="Model Performance"
                    maxRows={20}
                  />
                </CardContent>
              </Card>
            )}
            
            {!chartQuery.data?.figure && !leaderboardQuery.data?.leaderboard && (
              <Card>
                <CardContent className="flex items-center justify-center h-64">
                  <div className="text-center text-muted-foreground">
                    <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
                    <p>No analysis results available</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
