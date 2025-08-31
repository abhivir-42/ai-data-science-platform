'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowLeft, Target, Database, Upload, PlayCircle, ArrowRight, Settings, Brain, FileText, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { FileUploader } from '@/components/core/file-uploader'
import { ProgressIndicator } from '@/components/core/progress-indicator'
import { predictionClient } from '@/lib/uagent-client'
import { useSessionsStore } from '@/lib/store'
import { useToast } from '@/hooks/use-toast'
import type { PredictSingleParams, PredictBatchParams, AnalyzeModelParams } from '@/lib/uagent-client'

interface UploadedFileData {
  file: File
  id: string
  status: 'pending' | 'uploading' | 'completed' | 'error'
  progress?: number
  base64?: string
  error?: string
}

export function MLPredictionWorkspace() {
  const [activeTab, setActiveTab] = useState('single')
  const [modelSessionId, setModelSessionId] = useState('')
  const [modelPath, setModelPath] = useState('')
  const [singleInputData, setSingleInputData] = useState('')
  const [batchFiles, setBatchFiles] = useState<UploadedFileData[]>([])
  const [analysisQuery, setAnalysisQuery] = useState('')
  
  const router = useRouter()
  const { addSession, sessions } = useSessionsStore()
  const { toast } = useToast()

  // Single prediction mutation
  const predictSingleMutation = useMutation({
    mutationFn: async (params: PredictSingleParams) => {
      return await predictionClient.predictSingle(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        addSession({
          sessionId: response.session_id,
          agentType: 'prediction',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: `Single prediction using model ${modelSessionId.slice(0, 8)}...`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Prediction completed! 🎯",
          description: `Session ${response.session_id.slice(0, 8)}... created`,
          variant: "default",
        })

        router.push(`/sessions/${response.session_id}`)
      } else {
        throw new Error(response.error || 'Failed to make prediction')
      }
    },
    onError: (error) => {
      toast({
        title: "Prediction failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  // Batch prediction mutation
  const predictBatchMutation = useMutation({
    mutationFn: async (params: PredictBatchParams) => {
      return await predictionClient.predictBatch(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        addSession({
          sessionId: response.session_id,
          agentType: 'prediction',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: `Batch predictions using model ${modelSessionId.slice(0, 8)}...`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Batch predictions completed! 🎯",
          description: `Session ${response.session_id.slice(0, 8)}... created`,
          variant: "default",
        })

        router.push(`/sessions/${response.session_id}`)
      } else {
        throw new Error(response.error || 'Failed to make batch predictions')
      }
    },
    onError: (error) => {
      toast({
        title: "Batch prediction failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  // Model analysis mutation
  const analyzeModelMutation = useMutation({
    mutationFn: async (params: AnalyzeModelParams) => {
      return await predictionClient.analyzeModel(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        addSession({
          sessionId: response.session_id,
          agentType: 'prediction',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: `Model analysis: ${analysisQuery.slice(0, 50)}...`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Model analysis completed! 🧠",
          description: `Session ${response.session_id.slice(0, 8)}... created`,
          variant: "default",
        })

        router.push(`/sessions/${response.session_id}`)
      } else {
        throw new Error(response.error || 'Failed to analyze model')
      }
    },
    onError: (error) => {
      toast({
        title: "Model analysis failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  const handleBatchFileUpload = (files: UploadedFileData[]) => {
    setBatchFiles(files)
  }

  const handleSinglePrediction = () => {
    if (!modelSessionId.trim() && !modelPath.trim()) {
      toast({
        title: "Model required",
        description: "Please specify a model session ID or model path",
        variant: "destructive",
      })
      return
    }

    if (!singleInputData.trim()) {
      toast({
        title: "Input data required",
        description: "Please provide input data for prediction",
        variant: "destructive",
      })
      return
    }

    try {
      const inputData = JSON.parse(singleInputData)
      predictSingleMutation.mutate({
        model_session_id: modelSessionId || undefined,
        model_path: modelPath || undefined,
        input_data: inputData,
      })
    } catch (e) {
      toast({
        title: "Invalid input data",
        description: "Please provide valid JSON input data",
        variant: "destructive",
      })
    }
  }

  const handleBatchPrediction = () => {
    if (!modelSessionId.trim() && !modelPath.trim()) {
      toast({
        title: "Model required",
        description: "Please specify a model session ID or model path",
        variant: "destructive",
      })
      return
    }

    if (batchFiles.length === 0) {
      toast({
        title: "No files selected",
        description: "Please upload a file for batch prediction",
        variant: "destructive",
      })
      return
    }

    const file = batchFiles[0]
    predictBatchMutation.mutate({
      model_session_id: modelSessionId || undefined,
      model_path: modelPath || undefined,
      filename: file.file.name,
      file_content: file.base64,
    })
  }

  const handleModelAnalysis = () => {
    if (!modelSessionId.trim() && !modelPath.trim()) {
      toast({
        title: "Model required",
        description: "Please specify a model session ID or model path",
        variant: "destructive",
      })
      return
    }

    if (!analysisQuery.trim()) {
      toast({
        title: "Analysis query required",
        description: "Please specify what you'd like to know about the model",
        variant: "destructive",
      })
      return
    }

    analyzeModelMutation.mutate({
      model_session_id: modelSessionId || undefined,
      model_path: modelPath || undefined,
      query: analysisQuery,
    })
  }

  // Get recent sessions
  const recentSessions = sessions
    .filter(s => s.agentType === 'prediction')
    .slice(0, 5)

  const trainingSessionsForModels = sessions
    .filter(s => s.agentType === 'training')
    .slice(0, 10)

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-50 via-pink-50 to-rose-50">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2">
            {/* Header */}
            <div className="flex items-center gap-4 mb-8">
              <Link href="/">
                <Button variant="outline" size="sm">
                  <ArrowLeft className="h-4 w-4 mr-2" />
                  Back to Dashboard
                </Button>
              </Link>
              <div className="flex items-center gap-3">
                <div className="p-2 bg-red-100 rounded-lg">
                  <Target className="h-6 w-6 text-red-600" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">ML Prediction</h1>
                  <p className="text-gray-600">Make predictions using trained machine learning models</p>
                </div>
              </div>
            </div>

            {/* Model Selection */}
            <Card className="backdrop-blur-sm bg-white/80 border-white/20 shadow-xl mb-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-red-500" />
                  Model Selection
                </CardTitle>
                <CardDescription>
                  Choose a trained model for making predictions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <Label htmlFor="model-session">Model Session ID</Label>
                  <Input
                    id="model-session"
                    placeholder="Enter session ID from ML training..."
                    value={modelSessionId}
                    onChange={(e) => setModelSessionId(e.target.value)}
                  />
                </div>
                
                <div className="text-center text-sm text-muted-foreground">or</div>
                
                <div>
                  <Label htmlFor="model-path">Model File Path</Label>
                  <Input
                    id="model-path"
                    placeholder="Enter path to saved model file..."
                    value={modelPath}
                    onChange={(e) => setModelPath(e.target.value)}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Prediction Interface */}
            <Card className="backdrop-blur-sm bg-white/80 border-white/20 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="h-5 w-5 text-red-500" />
                  Prediction Operations
                </CardTitle>
                <CardDescription>
                  Make single predictions, batch predictions, or analyze your model
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="single" className="flex items-center gap-2">
                      <Target className="h-4 w-4" />
                      Single
                    </TabsTrigger>
                    <TabsTrigger value="batch" className="flex items-center gap-2">
                      <Upload className="h-4 w-4" />
                      Batch
                    </TabsTrigger>
                    <TabsTrigger value="analyze" className="flex items-center gap-2">
                      <BarChart3 className="h-4 w-4" />
                      Analyze
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="single" className="space-y-6 mt-6">
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="input-data">Input Data (JSON)</Label>
                        <Textarea
                          id="input-data"
                          placeholder='{"feature1": 25, "feature2": "category_a", "feature3": 1.5}'
                          value={singleInputData}
                          onChange={(e) => setSingleInputData(e.target.value)}
                          rows={6}
                        />
                        <p className="text-xs text-muted-foreground mt-1">
                          Provide input data as JSON object with feature names and values
                        </p>
                      </div>

                      <div className="flex justify-end">
                        <Button
                          onClick={handleSinglePrediction}
                          disabled={predictSingleMutation.isPending}
                          className="bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700"
                        >
                          {predictSingleMutation.isPending ? (
                            <>
                              <ProgressIndicator status="running" className="mr-2" />
                              Predicting...
                            </>
                          ) : (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" />
                              Make Prediction
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="batch" className="space-y-6 mt-6">
                    <div className="space-y-4">
                      <div>
                        <Label>Upload Data for Batch Prediction</Label>
                        <FileUploader
                          onUpload={handleBatchFileUpload}
                          accept={['csv', 'xlsx', 'json']}
                          maxFiles={1}
                          className="mt-2"
                        />
                        {batchFiles.length > 0 && (
                          <div className="mt-2 text-sm text-green-600">
                            ✓ {batchFiles[0].file.name} uploaded
                          </div>
                        )}
                        <p className="text-xs text-muted-foreground mt-1">
                          Upload a file with the same features used for training
                        </p>
                      </div>

                      <div className="flex justify-end">
                        <Button
                          onClick={handleBatchPrediction}
                          disabled={predictBatchMutation.isPending || batchFiles.length === 0}
                          className="bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700"
                        >
                          {predictBatchMutation.isPending ? (
                            <>
                              <ProgressIndicator status="running" className="mr-2" />
                              Processing Batch...
                            </>
                          ) : (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" />
                              Predict Batch
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="analyze" className="space-y-6 mt-6">
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="analysis-query">Analysis Question</Label>
                        <Textarea
                          id="analysis-query"
                          placeholder="What would you like to know about the model? (e.g., 'What are the most important features?', 'How accurate is this model?', 'Show me feature importance')"
                          value={analysisQuery}
                          onChange={(e) => setAnalysisQuery(e.target.value)}
                          rows={4}
                        />
                      </div>

                      <div className="flex justify-end">
                        <Button
                          onClick={handleModelAnalysis}
                          disabled={analyzeModelMutation.isPending}
                          className="bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700"
                        >
                          {analyzeModelMutation.isPending ? (
                            <>
                              <ProgressIndicator status="running" className="mr-2" />
                              Analyzing...
                            </>
                          ) : (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" />
                              Analyze Model
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Available Models */}
            {trainingSessionsForModels.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Brain className="h-4 w-4" />
                    Available Models
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {trainingSessionsForModels.map((session) => (
                      <div
                        key={session.sessionId}
                        className="flex items-center justify-between p-2 bg-gray-50 rounded cursor-pointer hover:bg-gray-100"
                        onClick={() => setModelSessionId(session.sessionId)}
                      >
                        <div className="text-sm">
                          <div className="font-medium">{session.sessionId.slice(0, 8)}...</div>
                          <div className="text-xs text-gray-500">{session.description}</div>
                        </div>
                        <ArrowRight className="h-4 w-4 text-gray-400" />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Prediction Types */}
            <Card>
              <CardHeader>
                <CardTitle>Prediction Types</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid gap-2">
                  {[
                    { type: 'Single', desc: 'One prediction at a time' },
                    { type: 'Batch', desc: 'Multiple predictions from file' },
                    { type: 'Analysis', desc: 'Model interpretation' },
                    { type: 'Features', desc: 'Feature importance' },
                    { type: 'Performance', desc: 'Model metrics' },
                    { type: 'Explanations', desc: 'Prediction reasoning' },
                  ].map(pred => (
                    <div key={pred.type} className="text-sm">
                      <div className="font-medium">{pred.type}</div>
                      <div className="text-xs text-muted-foreground">{pred.desc}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Recent Sessions */}
            {recentSessions.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recent Predictions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {recentSessions.map((session) => (
                      <div
                        key={session.sessionId}
                        className="flex items-center justify-between p-2 bg-gray-50 rounded cursor-pointer hover:bg-gray-100"
                        onClick={() => router.push(`/sessions/${session.sessionId}`)}
                      >
                        <div className="text-sm">
                          <div className="font-medium">{session.sessionId.slice(0, 8)}...</div>
                          <div className="text-xs text-gray-500">{session.description}</div>
                        </div>
                        <ArrowRight className="h-4 w-4 text-gray-400" />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Help */}
            <Card>
              <CardHeader>
                <CardTitle>Quick Help</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="text-sm space-y-2">
                  <p><strong>Single Prediction:</strong> Predict one instance using JSON input</p>
                  <p><strong>Batch Prediction:</strong> Predict multiple instances from a CSV file</p>
                  <p><strong>Model Analysis:</strong> Ask questions about your trained model</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}