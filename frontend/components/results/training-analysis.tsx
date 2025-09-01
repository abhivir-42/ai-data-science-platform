'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Brain, Code, FileText, Download, Clock, Database, Target, BarChart3, Zap } from 'lucide-react'
import { CodeViewer } from '@/components/core/code-viewer'

interface TrainingAnalysisProps {
  sessionId: string
  trainingFunction?: string
  workflowSummary?: any
  logs?: string[]
  mlSteps?: string[]
  originalData?: any
  bestModelId?: string
  modelPath?: string
  executionTime?: number
  className?: string
}

export function TrainingAnalysis({
  sessionId,
  trainingFunction,
  workflowSummary,
  logs,
  mlSteps,
  originalData,
  bestModelId,
  modelPath,
  executionTime,
  className
}: TrainingAnalysisProps) {
  const [activeTab, setActiveTab] = useState('overview')

  const formatTime = (seconds?: number) => {
    if (!seconds) return 'Unknown'
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.floor(seconds % 60)
    return minutes > 0 ? `${minutes}m ${remainingSeconds}s` : `${remainingSeconds}s`
  }

  const downloadCode = () => {
    if (!trainingFunction) return
    
    const blob = new Blob([trainingFunction], { type: 'text/python' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `h2o_training_${sessionId.slice(0, 8)}.py`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const downloadLogs = () => {
    if (!logs || logs.length === 0) return
    
    const logContent = logs.join('\n')
    const blob = new Blob([logContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `training_logs_${sessionId.slice(0, 8)}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-indigo-500" />
          Training Analysis & Results
        </CardTitle>
        <CardDescription>
          Comprehensive analysis of your H2O AutoML training session
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="code">Generated Code</TabsTrigger>
            <TabsTrigger value="workflow">Workflow</TabsTrigger>
            <TabsTrigger value="logs">Execution Logs</TabsTrigger>
            <TabsTrigger value="data">Training Data</TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-6 mt-6">
            {/* Training Summary */}
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-2">
                      <Clock className="h-8 w-8 text-blue-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {formatTime(executionTime)}
                    </div>
                    <div className="text-sm text-gray-600">Training Time</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-2">
                      <Target className="h-8 w-8 text-green-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {bestModelId ? '1' : '0'}
                    </div>
                    <div className="text-sm text-gray-600">Best Model</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-2">
                      <Database className="h-8 w-8 text-purple-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {originalData?.shape?.[0] || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Training Rows</div>
                  </div>
                  
                  <div className="text-center">
                    <div className="flex items-center justify-center mb-2">
                      <BarChart3 className="h-8 w-8 text-orange-500" />
                    </div>
                    <div className="text-2xl font-bold text-gray-900">
                      {originalData?.shape?.[1] || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-600">Features</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Best Model Info */}
            {bestModelId && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Zap className="h-5 w-5 text-yellow-500" />
                    Champion Model
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Model ID:</span>
                      <Badge variant="default" className="font-mono text-xs">
                        {bestModelId}
                      </Badge>
                    </div>
                    
                    {modelPath && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Saved Path:</span>
                        <span className="text-xs font-mono text-gray-600 max-w-md truncate">
                          {modelPath}
                        </span>
                      </div>
                    )}
                    
                    <div className="pt-2">
                      <Button size="sm" variant="outline" className="w-full">
                        <Download className="h-4 w-4 mr-2" />
                        Download Best Model
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* ML Steps Summary */}
            {mlSteps && mlSteps.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Recommended Next Steps</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {mlSteps.map((step, index) => (
                      <div key={index} className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg">
                        <div className="flex-shrink-0 w-6 h-6 bg-indigo-100 rounded-full flex items-center justify-center text-sm font-bold text-indigo-600">
                          {index + 1}
                        </div>
                        <div className="text-sm text-gray-700">{step}</div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="code" className="space-y-4 mt-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Generated H2O Training Code</h3>
                <p className="text-sm text-gray-600">
                  This Python function was automatically generated to reproduce your training
                </p>
              </div>
              {trainingFunction && (
                <Button onClick={downloadCode} size="sm" variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download Code
                </Button>
              )}
            </div>
            
            {trainingFunction ? (
              <CodeViewer 
                code={trainingFunction} 
                language="python"
                showLineNumbers={true}
                className="max-h-96"
              />
            ) : (
              <Card>
                <CardContent className="flex items-center justify-center h-32">
                  <div className="text-center text-gray-500">
                    <Code className="h-8 w-8 mx-auto mb-2" />
                    <p>No training code available</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="workflow" className="space-y-4 mt-6">
            <div>
              <h3 className="text-lg font-semibold mb-4">Training Workflow Summary</h3>
              
              {workflowSummary ? (
                <Card>
                  <CardContent className="p-6">
                    <pre className="whitespace-pre-wrap text-sm bg-gray-50 p-4 rounded-lg overflow-auto max-h-96">
                      {typeof workflowSummary === 'string' 
                        ? workflowSummary 
                        : JSON.stringify(workflowSummary, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="flex items-center justify-center h-32">
                    <div className="text-center text-gray-500">
                      <FileText className="h-8 w-8 mx-auto mb-2" />
                      <p>No workflow summary available</p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          <TabsContent value="logs" className="space-y-4 mt-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold">Training Execution Logs</h3>
                <p className="text-sm text-gray-600">
                  Detailed logs from the H2O AutoML training process
                </p>
              </div>
              {logs && logs.length > 0 && (
                <Button onClick={downloadLogs} size="sm" variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download Logs
                </Button>
              )}
            </div>
            
            {logs && logs.length > 0 ? (
              <Card>
                <CardContent className="p-6">
                  <div className="bg-black text-green-400 p-4 rounded-lg font-mono text-sm overflow-auto max-h-96">
                    {logs.map((log, index) => (
                      <div key={index} className="mb-1">
                        <span className="text-gray-500">[{index + 1}]</span> {log}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="flex items-center justify-center h-32">
                  <div className="text-center text-gray-500">
                    <FileText className="h-8 w-8 mx-auto mb-2" />
                    <p>No training logs available</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="data" className="space-y-4 mt-6">
            <div>
              <h3 className="text-lg font-semibold mb-4">Training Dataset Information</h3>
              
              {originalData ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Dataset Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-blue-600">
                          {originalData.shape?.[0] || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Rows</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-green-600">
                          {originalData.shape?.[1] || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Columns</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">
                          {originalData.columns?.length || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Features</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-600">
                          {originalData.columns?.filter((col: string) => 
                            originalData.records?.[0]?.[col] && 
                            typeof originalData.records[0][col] === 'number'
                          ).length || 'N/A'}
                        </div>
                        <div className="text-sm text-gray-600">Numeric</div>
                      </div>
                    </div>
                    
                    {originalData.columns && (
                      <div>
                        <h4 className="font-semibold mb-2">Column Names:</h4>
                        <div className="flex flex-wrap gap-2">
                          {originalData.columns.map((col: string, index: number) => (
                            <Badge key={index} variant="outline" className="text-xs">
                              {col}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="flex items-center justify-center h-32">
                    <div className="text-center text-gray-500">
                      <Database className="h-8 w-8 mx-auto mb-2" />
                      <p>No training data information available</p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
