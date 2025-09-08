'use client'

import React, { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import Link from 'next/link'
import { 
  ArrowLeft, Download, RefreshCw, Clock, CheckCircle, AlertCircle, 
  Database, Code, BarChart3, FileText, Lightbulb, TrendingUp, 
  Copy, Zap, Target, Award, Brain, Sparkles, ChevronDown, ChevronUp
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { DataFrameViewer } from '@/components/core/dataframe-viewer'
import { CodeViewer } from '@/components/core/code-viewer'
import { PlotlyChart } from '@/components/core/plotly-chart'
import { WorkflowClient, type WorkflowResultsResponse } from '@/lib/workflow-client'
import { cn, formatDuration, formatRelativeTime } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'

interface WorkflowResultsViewerProps {
  workflowId: string
}

const agentConfig = {
  loading: { 
    title: 'Data Loading', 
    color: 'blue', 
    icon: Database,
    description: 'Data ingestion and initial processing'
  },
  cleaning: { 
    title: 'Data Cleaning', 
    color: 'green', 
    icon: Sparkles,
    description: 'Data quality improvement and preparation'
  },
  visualization: { 
    title: 'Data Visualization', 
    color: 'purple', 
    icon: BarChart3,
    description: 'Chart generation and visual insights'
  },
  engineering: { 
    title: 'Feature Engineering', 
    color: 'orange', 
    icon: Brain,
    description: 'Feature creation and transformation'
  },
  training: { 
    title: 'ML Training', 
    color: 'indigo', 
    icon: Target,
    description: 'Machine learning model training'
  },
  prediction: { 
    title: 'ML Prediction', 
    color: 'red', 
    icon: Zap,
    description: 'Model inference and predictions'
  },
} as const

export function WorkflowResultsViewer({ workflowId }: WorkflowResultsViewerProps) {
  const [activeTab, setActiveTab] = useState('overview')
  const [expandedSteps, setExpandedSteps] = useState<Record<string, boolean>>({})
  const { toast } = useToast()

  const workflowClient = new WorkflowClient()

  // Fetch workflow results
  const { data: workflowResults, isLoading, error, refetch } = useQuery({
    queryKey: ['workflow-results', workflowId],
    queryFn: () => workflowClient.getWorkflowResults(workflowId),
    refetchInterval: false,
    retry: 3,
  })

  const toggleStepExpansion = (stepId: string) => {
    setExpandedSteps(prev => ({
      ...prev,
      [stepId]: !prev[stepId]
    }))
  }

  const handleDownloadResults = async () => {
    if (!workflowResults) return

    try {
      const dataStr = JSON.stringify(workflowResults, null, 2)
      const dataBlob = new Blob([dataStr], { type: 'application/json' })
      const url = URL.createObjectURL(dataBlob)
      const link = document.createElement('a')
      link.href = url
      link.download = `workflow-results-${workflowId}.json`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)

      toast({
        title: "Results Downloaded",
        description: "Workflow results have been downloaded successfully",
      })
    } catch (error) {
      toast({
        title: "Download Failed",
        description: "Failed to download workflow results",
        variant: "destructive",
      })
    }
  }

  const copyToClipboard = async (text: string, label: string) => {
    try {
      await navigator.clipboard.writeText(text)
      toast({
        title: "Copied!",
        description: `${label} copied to clipboard`,
      })
    } catch (error) {
      toast({
        title: "Copy Failed",
        description: "Failed to copy to clipboard",
        variant: "destructive",
      })
    }
  }

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="space-y-2">
            <div className="h-8 w-64 bg-gray-200 rounded animate-pulse" />
            <div className="h-4 w-96 bg-gray-200 rounded animate-pulse" />
          </div>
          <div className="h-10 w-32 bg-gray-200 rounded animate-pulse" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-24 bg-gray-200 rounded animate-pulse" />
          ))}
        </div>
        <div className="h-96 bg-gray-200 rounded animate-pulse" />
      </div>
    )
  }

  if (error || !workflowResults) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] space-y-4">
        <AlertCircle className="h-16 w-16 text-red-500" />
        <h2 className="text-2xl font-semibold">Failed to Load Results</h2>
        <p className="text-gray-600 text-center max-w-md">
          We could not load the workflow results. This might be because the workflow is still running or there was an error.
        </p>
        <div className="flex gap-4">
          <Button onClick={() => refetch()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Retry
          </Button>
          <Link href="/workflows">
            <Button variant="outline">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Workflows
            </Button>
          </Link>
        </div>
      </div>
    )
  }

  const results = workflowResults.results
  const stepResults = results?.step_results || []
  const completedSteps = workflowResults.steps?.filter(s => s.status === 'completed') || []
  const totalExecutionTime = workflowResults.total_execution_time || 0

  // Extract key metrics
  const dataMetrics = {
    originalRows: 0,
    finalRows: 0,
    columnsProcessed: 0,
    dataQualityScore: 0,
    visualizationsCreated: 0,
    codeFilesGenerated: 0
  }

  // Process step results to extract metrics
  stepResults.forEach((step: any) => {
    if (step.agent_type === 'loading' && step.data_info?.data) {
      dataMetrics.originalRows = step.data_info.data.shape?.[0] || 0
      dataMetrics.columnsProcessed = step.data_info.data.shape?.[1] || 0
    }
    if (step.agent_type === 'cleaning' && step.cleaned_data?.data) {
      dataMetrics.finalRows = step.cleaned_data.data.shape?.[0] || 0
      dataMetrics.dataQualityScore = Math.round(((dataMetrics.finalRows / Math.max(dataMetrics.originalRows, 1)) * 100))
    }
    if (step.agent_type === 'visualization') {
      dataMetrics.visualizationsCreated = step.chart ? 1 : 0
    }
    if (step.cleaning_code || step.viz_code) {
      dataMetrics.codeFilesGenerated += 1
    }
  })

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-4">
            <Link href="/workflows">
              <Button variant="ghost" size="sm">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Workflows
              </Button>
            </Link>
            <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
              <CheckCircle className="mr-1 h-3 w-3" />
              Completed
            </Badge>
          </div>
          <h1 className="text-3xl font-bold text-gray-900">{workflowResults.name}</h1>
          <p className="text-gray-600 flex items-center gap-4">
            <span className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              {formatDuration(totalExecutionTime)}
            </span>
            <span className="flex items-center gap-1">
              <CheckCircle className="h-4 w-4" />
              {completedSteps.length} steps completed
            </span>
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => refetch()} variant="outline" size="sm">
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
          <Button onClick={handleDownloadResults} size="sm">
            <Download className="mr-2 h-4 w-4" />
            Download Results
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Data Rows</p>
                <p className="text-2xl font-bold">{dataMetrics.originalRows.toLocaleString()}</p>
              </div>
              <Database className="h-8 w-8 text-blue-500" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Data Quality</p>
                <p className="text-2xl font-bold">{dataMetrics.dataQualityScore}%</p>
              </div>
              <Sparkles className="h-8 w-8 text-green-500" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Visualizations</p>
                <p className="text-2xl font-bold">{dataMetrics.visualizationsCreated}</p>
              </div>
              <BarChart3 className="h-8 w-8 text-purple-500" />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">Code Generated</p>
                <p className="text-2xl font-bold">{dataMetrics.codeFilesGenerated}</p>
              </div>
              <Code className="h-8 w-8 text-orange-500" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="data">Data Journey</TabsTrigger>
          <TabsTrigger value="code">Generated Code</TabsTrigger>
          <TabsTrigger value="visualizations">Charts</TabsTrigger>
          <TabsTrigger value="insights">Insights</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="h-5 w-5" />
                Workflow Execution Summary
              </CardTitle>
              <CardDescription>
                Complete overview of your data analysis workflow
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Execution Timeline */}
                <div className="space-y-4">
                  <h4 className="font-semibold">Execution Timeline</h4>
                  <div className="space-y-3">
                    {workflowResults.steps?.map((step, index) => {
                      const config = agentConfig[step.agent_type as keyof typeof agentConfig]
                      const Icon = config?.icon || Database
                      const stepResult = stepResults.find((r: any) => r.agent_type === step.agent_type)
                      
                      return (
                        <div key={step.id} className="flex items-center gap-4 p-4 border rounded-lg">
                          <div className={cn(
                            "flex items-center justify-center w-10 h-10 rounded-full",
                            step.status === 'completed' ? 'bg-green-100 text-green-600' : 'bg-gray-100 text-gray-400'
                          )}>
                            <Icon className="h-5 w-5" />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center justify-between">
                              <h5 className="font-medium">{config?.title || step.agent_type}</h5>
                              <Badge variant={step.status === 'completed' ? 'default' : 'secondary'}>
                                {step.status}
                              </Badge>
                            </div>
                            <p className="text-sm text-gray-600">{config?.description}</p>
                            {step.execution_time_seconds && (
                              <p className="text-xs text-gray-500 mt-1">
                                Completed in {formatDuration(step.execution_time_seconds)}
                              </p>
                            )}
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleStepExpansion(step.id)}
                          >
                            {expandedSteps[step.id] ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* Key Achievements */}
                <div className="space-y-4">
                  <h4 className="font-semibold">Key Achievements</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <Alert>
                      <Award className="h-4 w-4" />
                      <AlertTitle>Data Processing</AlertTitle>
                      <AlertDescription>
                        Successfully processed {dataMetrics.originalRows.toLocaleString()} rows 
                        across {dataMetrics.columnsProcessed} columns with {dataMetrics.dataQualityScore}% data retention
                      </AlertDescription>
                    </Alert>
                    <Alert>
                      <Lightbulb className="h-4 w-4" />
                      <AlertTitle>Code Generation</AlertTitle>
                      <AlertDescription>
                        Generated {dataMetrics.codeFilesGenerated} production-ready code files 
                        for data processing and visualization
                      </AlertDescription>
                    </Alert>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data Journey Tab */}
        <TabsContent value="data" className="space-y-6">
          {stepResults.map((stepResult: any, index: number) => (
            <Card key={index}>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {React.createElement(agentConfig[stepResult.agent_type as keyof typeof agentConfig]?.icon || Database, { className: "h-5 w-5" })}
                  {agentConfig[stepResult.agent_type as keyof typeof agentConfig]?.title || stepResult.agent_type}
                </CardTitle>
                <CardDescription>
                  Step {index + 1} of {stepResults.length} - Data transformation and processing
                </CardDescription>
              </CardHeader>
              <CardContent>
                {/* Loading Agent Data */}
                {stepResult.agent_type === 'loading' && stepResult.data_info?.data && (
                  <div className="space-y-4">
                    <h5 className="font-medium">Original Dataset</h5>
                    <DataFrameViewer 
                      data={stepResult.data_info.data} 
                      title="Loaded Data"
                      maxRows={10}
                    />
                    <div className="flex gap-4 text-sm text-gray-600">
                      <span>Rows: {stepResult.data_info.data.shape?.[0] || 0}</span>
                      <span>Columns: {stepResult.data_info.data.shape?.[1] || 0}</span>
                    </div>
                  </div>
                )}

                {/* Cleaning Agent Data */}
                {stepResult.agent_type === 'cleaning' && stepResult.cleaned_data?.data && (
                  <div className="space-y-4">
                    <h5 className="font-medium">Cleaned Dataset</h5>
                    <DataFrameViewer 
                      data={stepResult.cleaned_data.data} 
                      title="Cleaned Data"
                      maxRows={10}
                    />
                    <div className="flex gap-4 text-sm text-gray-600">
                      <span>Rows: {stepResult.cleaned_data.data.shape?.[0] || 0}</span>
                      <span>Columns: {stepResult.cleaned_data.data.shape?.[1] || 0}</span>
                      <span className="text-green-600">
                        Quality: {Math.round(((stepResult.cleaned_data.data.shape?.[0] || 0) / Math.max(dataMetrics.originalRows, 1)) * 100)}%
                      </span>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </TabsContent>

        {/* Generated Code Tab */}
        <TabsContent value="code" className="space-y-6">
          {stepResults.map((stepResult: any, index: number) => {
            const hasCode = stepResult.cleaning_code || stepResult.viz_code
            if (!hasCode) return null

            return (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Code className="h-5 w-5" />
                      {agentConfig[stepResult.agent_type as keyof typeof agentConfig]?.title || stepResult.agent_type} Code
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => copyToClipboard(
                        stepResult.cleaning_code?.generated_code || stepResult.viz_code?.generated_code || '',
                        'Code'
                      )}
                    >
                      <Copy className="mr-2 h-4 w-4" />
                      Copy
                    </Button>
                  </CardTitle>
                  <CardDescription>
                    Production-ready code generated by the {stepResult.agent_type} agent
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {stepResult.cleaning_code && (
                    <div className="space-y-4">
                      <h5 className="font-medium">Data Cleaning Function</h5>
                      <CodeViewer
                        code={stepResult.cleaning_code.generated_code}
                        language="python"
                        title="data_cleaner.py"
                      />
                      {stepResult.cleaning_code.code_explanation && (
                        <Alert>
                          <FileText className="h-4 w-4" />
                          <AlertTitle>Code Explanation</AlertTitle>
                          <AlertDescription>{stepResult.cleaning_code.code_explanation}</AlertDescription>
                        </Alert>
                      )}
                    </div>
                  )}
                  
                  {stepResult.viz_code && (
                    <div className="space-y-4">
                      <h5 className="font-medium">Visualization Function</h5>
                      <CodeViewer
                        code={stepResult.viz_code.generated_code}
                        language="python"
                        title="data_visualization.py"
                      />
                      {stepResult.viz_code.code_explanation && (
                        <Alert>
                          <FileText className="h-4 w-4" />
                          <AlertTitle>Code Explanation</AlertTitle>
                          <AlertDescription>{stepResult.viz_code.code_explanation}</AlertDescription>
                        </Alert>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </TabsContent>

        {/* Visualizations Tab */}
        <TabsContent value="visualizations" className="space-y-6">
          {stepResults.map((stepResult: any, index: number) => {
            if (stepResult.agent_type !== 'visualization' || !stepResult.chart) return null

            return (
              <Card key={index}>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5" />
                    Generated Visualizations
                  </CardTitle>
                  <CardDescription>
                    Interactive charts and graphs created from your data
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {stepResult.chart?.plotly_chart ? (
                    <PlotlyChart
                      figure={stepResult.chart.plotly_chart}
                      title="Data Analysis Visualization"
                      description="AI-generated chart with insights from your data"
                      height={500}
                    />
                  ) : stepResult.viz_code?.plotly_chart ? (
                    <PlotlyChart
                      figure={stepResult.viz_code.plotly_chart}
                      title="Data Analysis Visualization"
                      description="AI-generated chart with insights from your data"
                      height={500}
                    />
                  ) : (
                    <div className="text-center py-8 text-gray-500">
                      <BarChart3 className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                      <p>Chart data not available</p>
                      <p className="text-sm mt-2">
                        Chart generation: {stepResult.chart?.success ? 'Success' : 'Failed'}
                        {stepResult.chart?.message && (
                          <span className="block text-xs text-gray-400 mt-1">
                            {stepResult.chart.message}
                          </span>
                        )}
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </TabsContent>

        {/* Insights Tab */}
        <TabsContent value="insights" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Lightbulb className="h-5 w-5" />
                Key Insights & Recommendations
              </CardTitle>
              <CardDescription>
                AI-generated insights and next steps based on your data analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-4">
                    <h5 className="font-semibold">Data Quality Insights</h5>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm">Data Completeness</span>
                        <span className="text-sm font-medium">{dataMetrics.dataQualityScore}%</span>
                      </div>
                      <Progress value={dataMetrics.dataQualityScore} className="h-2" />
                    </div>
                    <Alert>
                      <CheckCircle className="h-4 w-4" />
                      <AlertTitle>Quality Assessment</AlertTitle>
                      <AlertDescription>
                        Your data has {dataMetrics.dataQualityScore >= 90 ? 'excellent' : 
                                      dataMetrics.dataQualityScore >= 70 ? 'good' : 'acceptable'} quality 
                        with minimal data loss during cleaning.
                      </AlertDescription>
                    </Alert>
                  </div>

                  <div className="space-y-4">
                    <h5 className="font-semibold">Next Steps</h5>
                    <div className="space-y-2">
                      <Alert>
                        <Target className="h-4 w-4" />
                        <AlertTitle>Recommended Actions</AlertTitle>
                        <AlertDescription>
                          <ul className="list-disc list-inside space-y-1 mt-2">
                            <li>Download the generated code for production use</li>
                            <li>Explore the visualizations for deeper insights</li>
                            <li>Consider feature engineering for ML models</li>
                            <li>Schedule regular data quality monitoring</li>
                          </ul>
                        </AlertDescription>
                      </Alert>
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="space-y-4">
                  <h5 className="font-semibold">Technical Summary</h5>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="font-medium">Total Execution Time</span>
                      <p className="text-gray-600">{formatDuration(totalExecutionTime)}</p>
                    </div>
                    <div>
                      <span className="font-medium">Steps Completed</span>
                      <p className="text-gray-600">{completedSteps.length} of {workflowResults.steps?.length || 0}</p>
                    </div>
                    <div>
                      <span className="font-medium">Data Processed</span>
                      <p className="text-gray-600">{dataMetrics.originalRows.toLocaleString()} rows</p>
                    </div>
                    <div>
                      <span className="font-medium">Artifacts Created</span>
                      <p className="text-gray-600">{dataMetrics.codeFilesGenerated + dataMetrics.visualizationsCreated} files</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
