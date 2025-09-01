'use client'

import { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { ArrowLeft, RefreshCw, Download, ExternalLink, Database, Code, FileText, Lightbulb, BarChart3, Trophy, Brain, Target, Clock, Zap, TrendingUp, Award } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { DataFrameViewer } from '@/components/core/dataframe-viewer'
import { CodeViewer } from '@/components/core/code-viewer'
import { PlotlyChart } from '@/components/core/plotly-chart'
import { LeaderboardViewer } from '@/components/results/leaderboard-viewer'
import { TrainingAnalysis } from '@/components/results/training-analysis'
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
  const [detectedAgentType, setDetectedAgentType] = useState<AgentType | null>(null)
  const [isDetecting, setIsDetecting] = useState(false)
  const { getSession } = useSessionsStore()
  const { toast } = useToast()
  
  const session = getSession(sessionId)

  // Auto-detect agent type by trying training endpoints if session not found
  useEffect(() => {
    if (!session && !detectedAgentType && !isDetecting) {
      setIsDetecting(true)
      const tryDetectTraining = async () => {
        try {
          const trainingClient = getAgentClient('training')
          const leaderboard = await trainingClient.getSessionLeaderboard(sessionId)
          if (leaderboard.leaderboard) {
            setDetectedAgentType('training')
          }
        } catch (error) {
          // Try other agent types if needed
          console.log('Session not detected as training type')
        } finally {
          setIsDetecting(false)
        }
      }
      tryDetectTraining()
    }
  }, [session, sessionId, detectedAgentType, isDetecting])

  // Use detected session or fallback for unknown sessions
  const effectiveSession = session || (detectedAgentType ? {
    sessionId,
    agentType: detectedAgentType,
    createdAt: new Date().toISOString(),
    status: 'completed' as const,
    description: `${detectedAgentType} session`,
  } : null)

  // Get the appropriate client for this session
  const agentClient = effectiveSession ? getAgentClient(effectiveSession.agentType) : null

  // Queries for different session results
  const dataQuery = useQuery({
    queryKey: ['session-data', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      
      try {
        // Try different data endpoints based on agent type
        if (effectiveSession?.agentType === 'cleaning') {
          return await agentClient.getCleanedData(sessionId)
        } else {
          return await agentClient.getSessionData(sessionId)
        }
      } catch (error) {
        console.warn('Data not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!effectiveSession,
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
    enabled: !!agentClient && !!effectiveSession && ['cleaning', 'visualization', 'engineering', 'training'].includes(effectiveSession.agentType),
    retry: 1,
    staleTime: 30000,
  })

  const chartQuery = useQuery({
    queryKey: ['session-chart', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      return await agentClient.getSessionChart(sessionId)
    },
    enabled: !!agentClient && !!effectiveSession && effectiveSession.agentType === 'visualization',
    retry: 1,
    staleTime: 30000,
  })

  const leaderboardQuery = useQuery({
    queryKey: ['session-leaderboard', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      return await agentClient.getSessionLeaderboard(sessionId)
    },
    enabled: !!agentClient && !!effectiveSession && effectiveSession.agentType === 'training',
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
    enabled: !!agentClient && !!effectiveSession,
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
    enabled: !!agentClient && !!effectiveSession,
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
    enabled: !!agentClient && !!effectiveSession && effectiveSession.agentType === 'prediction',
    retry: 1,
    staleTime: 30000,
  })

  // 🔥 ML TRAINING SPECIFIC QUERIES 🔥
  const trainingFunctionQuery = useQuery({
    queryKey: ['training-function', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      try {
        return await (agentClient as any).getTrainingFunction(sessionId)
      } catch (error) {
        console.warn('Training function not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!effectiveSession && effectiveSession.agentType === 'training',
    retry: 1,
    staleTime: 30000,
  })

  const workflowSummaryQuery = useQuery({
    queryKey: ['workflow-summary', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      try {
        return await (agentClient as any).getWorkflowSummary(sessionId)
      } catch (error) {
        console.warn('Workflow summary not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!effectiveSession && effectiveSession.agentType === 'training',
    retry: 1,
    staleTime: 30000,
  })

  const bestModelQuery = useQuery({
    queryKey: ['best-model', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      try {
        return await (agentClient as any).getBestModelId(sessionId)
      } catch (error) {
        console.warn('Best model not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!effectiveSession && effectiveSession.agentType === 'training',
    retry: 1,
    staleTime: 30000,
  })

  const modelPathQuery = useQuery({
    queryKey: ['model-path', sessionId],
    queryFn: async () => {
      if (!agentClient) throw new Error('No agent client available')
      try {
        return await (agentClient as any).getModelPath(sessionId)
      } catch (error) {
        console.warn('Model path not available:', error)
        return null
      }
    },
    enabled: !!agentClient && !!effectiveSession && effectiveSession.agentType === 'training',
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
    
    // ML Training specific queries
    trainingFunctionQuery.refetch()
    workflowSummaryQuery.refetch()
    bestModelQuery.refetch()
    modelPathQuery.refetch()
    
    toast({
      title: "Refreshed",
      description: "Session results have been refreshed.",
      variant: "default",
    })
  }

  // Show loading state while detecting session type
  if (!effectiveSession && isDetecting) {
    return (
      <div className="container mx-auto px-4 py-8">
        <Card>
          <CardContent className="flex items-center justify-center h-64">
            <div className="text-center">
              <div className="text-4xl mb-4">🔍</div>
              <h3 className="text-lg font-medium mb-2">Detecting Session Type...</h3>
              <p className="text-muted-foreground">
                Looking for session {sessionId.slice(0, 8)}...
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Show not found only after detection is complete
  if (!effectiveSession && !isDetecting) {
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

  // At this point, effectiveSession is guaranteed to be non-null due to early returns above
  const agentInfo = agentConfig[effectiveSession!.agentType]

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
                      effectiveSession!.status === 'completed' ? 'secondary' : 
                      effectiveSession!.status === 'failed' ? 'destructive' : 'default'
                    }
                  >
                    {effectiveSession!.status}
                  </Badge>
                </div>
                
                <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                  <span>{agentInfo.title}</span>
                  <span>•</span>
                  <span>Session {sessionId.slice(0, 8)}...</span>
                  <span>•</span>
                  <span>{formatRelativeTime(effectiveSession!.createdAt)}</span>
                  {effectiveSession!.executionTimeSeconds && (
                    <>
                      <span>•</span>
                      <span>{formatDuration(effectiveSession!.executionTimeSeconds)}</span>
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
              
              <Link href={`/agents/${effectiveSession!.agentType}`}>
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
        {effectiveSession!.status === 'running' && (
          <div className="mb-6">
            <ProgressIndicator
              status="running"
              message="Processing your request..."
              startTime={effectiveSession!.createdAt}
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
                onDownload={(format) => {
                  if (dataQuery.data?.data) {
                    const data = dataQuery.data.data;
                    let content: string;
                    let filename: string;
                    let mimeType: string;
                    
                    if (format === 'csv') {
                      // Convert to CSV
                      const headers = data.columns.join(',');
                      const rows = data.records.map(record => 
                        data.columns.map(col => {
                          const value = record[col];
                          // Handle values with commas, quotes, or newlines
                          if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
                            return `"${value.replace(/"/g, '""')}"`;
                          }
                          return value ?? '';
                        }).join(',')
                      ).join('\n');
                      content = headers + '\n' + rows;
                      filename = `${session?.agentType || 'data'}_${sessionId.slice(0, 8)}_cleaned.csv`;
                      mimeType = 'text/csv';
                    } else {
                      // Convert to JSON
                      content = JSON.stringify(data, null, 2);
                      filename = `${session?.agentType || 'data'}_${sessionId.slice(0, 8)}_cleaned.json`;
                      mimeType = 'application/json';
                    }
                    
                    // Create and trigger download
                    const blob = new Blob([content], { type: mimeType });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    
                    toast({
                      title: "Download Complete",
                      description: `${filename} has been downloaded successfully`,
                    });
                  }
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
                downloadFileName={`${effectiveSession!.agentType}_${sessionId.slice(0, 8)}`}
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
            {effectiveSession!.agentType === 'visualization' && chartQuery.data?.figure ? (
              <PlotlyChart
                figure={chartQuery.data.figure as any}
                title="Generated Visualization"
                description="Interactive chart created by the visualization agent"
              />
            ) : null}
            
            {effectiveSession!.agentType === 'training' && (
              <div className="space-y-6">
                {/* 🎯 TRAINING OVERVIEW CARD */}
                <Card className="border-emerald-200 bg-gradient-to-br from-emerald-50 to-teal-50">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Brain className="h-6 w-6 text-emerald-600" />
                      <span>H2O AutoML Training Results</span>
                      <Badge variant="secondary" className="bg-emerald-100 text-emerald-700">
                        {effectiveSession!.status === 'completed' ? 'Completed' : 'Training'}
                      </Badge>
                    </CardTitle>
                    <CardDescription>
                      Automated machine learning with comprehensive model comparison and analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                      <div className="text-center p-4 rounded-lg bg-emerald-100">
                        <div className="text-2xl font-bold text-emerald-600">
                          {leaderboardQuery.data?.leaderboard 
                            ? (Array.isArray(leaderboardQuery.data.leaderboard) 
                               ? leaderboardQuery.data.leaderboard.length 
                               : Object.keys((leaderboardQuery.data.leaderboard as any)?.model_id || {}).length)
                            : 0}
                        </div>
                        <div className="text-sm text-emerald-700 flex items-center justify-center gap-1">
                          <Trophy className="h-3 w-3" />
                          Models Trained
                        </div>
                      </div>
                      <div className="text-center p-4 rounded-lg bg-blue-100">
                        <div className="text-2xl font-bold text-blue-600">
                          {bestModelQuery.data?.model_id ? '✓' : '⏳'}
                        </div>
                        <div className="text-sm text-blue-700 flex items-center justify-center gap-1">
                          <Target className="h-3 w-3" />
                          Best Model
                        </div>
                      </div>
                      <div className="text-center p-4 rounded-lg bg-purple-100">
                        <div className="text-2xl font-bold text-purple-600">
                          {effectiveSession!.executionTimeSeconds ? `${(effectiveSession!.executionTimeSeconds / 60).toFixed(1)}m` : '-'}
                        </div>
                        <div className="text-sm text-purple-700 flex items-center justify-center gap-1">
                          <Clock className="h-3 w-3" />
                          Training Time
                        </div>
                      </div>
                      <div className="text-center p-4 rounded-lg bg-amber-100">
                        <div className="text-2xl font-bold text-amber-600">
                          AutoML
                        </div>
                        <div className="text-sm text-amber-700 flex items-center justify-center gap-1">
                          <Zap className="h-3 w-3" />
                          H2O Engine
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* 🏆 ENHANCED LEADERBOARD COMPONENT */}
                {leaderboardQuery.data?.leaderboard && (
                  <LeaderboardViewer
                    leaderboard={Array.isArray(leaderboardQuery.data.leaderboard) 
                      ? leaderboardQuery.data.leaderboard 
                      : (() => {
                          // Transform H2O object format to array format for LeaderboardViewer
                          const data = leaderboardQuery.data.leaderboard as any;
                          if (data?.model_id && typeof data.model_id === 'object') {
                            return Object.keys(data.model_id).map(key => {
                              const entry: any = { model_id: data.model_id[key] };
                              // Add all metrics for this model
                              Object.keys(data).forEach(metric => {
                                if (metric !== 'model_id' && data[metric] && typeof data[metric] === 'object') {
                                  entry[metric] = data[metric][key];
                                }
                              });
                              return entry;
                            });
                          }
                          return [data];
                        })()}
                    bestModelId={bestModelQuery.data?.model_id}
                    onModelSelect={(modelId) => {
                      toast({
                        title: "Model Selected",
                        description: `Selected model: ${modelId}`,
                      })
                    }}
                    onDownloadModel={(modelId) => {
                      toast({
                        title: "Model Download",
                        description: `Preparing download for model: ${modelId}`,
                      })
                    }}
                  />
                )}
                
                {/* 🧠 COMPREHENSIVE TRAINING ANALYSIS */}
                <TrainingAnalysis
                  sessionId={sessionId}
                  trainingFunction={trainingFunctionQuery.data?.generated_code}
                  workflowSummary={workflowSummaryQuery.data?.data}
                  logs={logsQuery.data?.logs}
                  mlSteps={recommendationsQuery.data?.ml_steps || recommendationsQuery.data?.recommendations}
                  originalData={dataQuery.data?.data}
                  bestModelId={bestModelQuery.data?.model_id}
                  modelPath={modelPathQuery.data?.model_path}
                  executionTime={effectiveSession!.executionTimeSeconds}
                />
              </div>
            )}
            
            {!chartQuery.data?.figure && !leaderboardQuery.data?.leaderboard && effectiveSession!.agentType !== 'training' && (
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
