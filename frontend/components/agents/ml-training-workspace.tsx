'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowLeft, Brain, Database, Upload, PlayCircle, ArrowRight, Settings, Target, Clock, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { FileUploader } from '@/components/core/file-uploader'
import { ProgressIndicator } from '@/components/core/progress-indicator'
import { trainingClient } from '@/lib/uagent-client'
import { useSessionsStore } from '@/lib/store'
import { useToast } from '@/hooks/use-toast'
import type { TrainModelParams } from '@/lib/uagent-client'

interface TrainingOptions {
  max_runtime_secs?: number
  cv_folds?: number
  balance_classes?: boolean
  max_models?: number
  exclude_algos?: string[]
  seed?: number
}

interface UploadedFileData {
  file: File
  id: string
  status: 'pending' | 'uploading' | 'completed' | 'error'
  progress?: number
  base64?: string
  error?: string
}

export function MLTrainingWorkspace() {
  const [activeTab, setActiveTab] = useState('session')
  const [sessionId, setSessionId] = useState('')
  const [instructions, setInstructions] = useState('')
  const [targetVariable, setTargetVariable] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFileData[]>([])
  const [trainingOptions, setTrainingOptions] = useState<TrainingOptions>({
    max_runtime_secs: 300, // 5 minutes default
    cv_folds: 5,
    balance_classes: true,
    max_models: 20,
    exclude_algos: [],
    seed: 42,
  })
  
  const router = useRouter()
  const { addSession, sessions } = useSessionsStore()
  const { toast } = useToast()

  // ML training mutation
  const trainModelMutation = useMutation({
    mutationFn: async (params: TrainModelParams) => {
      return await trainingClient.trainModel(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        // Add session to store
        addSession({
          sessionId: response.session_id,
          agentType: 'training',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: sessionId 
            ? `Trained model on session ${sessionId.slice(0, 8)}... (target: ${targetVariable})`
            : `Trained model on uploaded data (target: ${targetVariable})`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Model training completed! 🤖",
          description: `Session ${response.session_id.slice(0, 8)}... created. View leaderboard in results.`,
          variant: "default",
        })

        // Navigate to session results
        router.push(`/sessions/${response.session_id}`)
      } else {
        throw new Error(response.error || 'Failed to train model')
      }
    },
    onError: (error) => {
      toast({
        title: "Model training failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  const handleFileUpload = (files: UploadedFileData[]) => {
    setUploadedFiles(files)
  }

  const handleTrainFromSession = () => {
    if (!sessionId.trim()) {
      toast({
        title: "Session ID required",
        description: "Please enter a session ID",
        variant: "destructive",
      })
      return
    }

    if (!targetVariable.trim()) {
      toast({
        title: "Target variable required",
        description: "Please specify the target variable for model training",
        variant: "destructive",
      })
      return
    }

    trainModelMutation.mutate({
      session_id: sessionId,
      target_variable: targetVariable,
      user_instructions: instructions || undefined,
      ...trainingOptions,
    })
  }

  const handleTrainFromFile = () => {
    if (uploadedFiles.length === 0) {
      toast({
        title: "No files selected",
        description: "Please upload a file before proceeding",
        variant: "destructive",
      })
      return
    }

    if (!targetVariable.trim()) {
      toast({
        title: "Target variable required",
        description: "Please specify the target variable for model training",
        variant: "destructive",
      })
      return
    }

    // Use the first file
    const file = uploadedFiles[0]
    trainModelMutation.mutate({
      filename: file.file.name,
      file_content: file.base64,
      target_variable: targetVariable,
      user_instructions: instructions || undefined,
      ...trainingOptions,
    })
  }

  // Get recent sessions for this agent
  const recentSessions = sessions
    .filter(s => s.agentType === 'training')
    .slice(0, 5)

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-blue-50 to-purple-50">
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
                <div className="p-2 bg-indigo-100 rounded-lg">
                  <Brain className="h-6 w-6 text-indigo-600" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">ML Training</h1>
                  <p className="text-gray-600">Train machine learning models with H2O AutoML</p>
                </div>
              </div>
            </div>

            {/* Main Interface */}
            <Card className="backdrop-blur-sm bg-white/80 border-white/20 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-indigo-500" />
                  Model Training Configuration
                </CardTitle>
                <CardDescription>
                  Train multiple ML models automatically and get a performance leaderboard
                </CardDescription>
              </CardHeader>
              <CardContent>
                <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                  <TabsList className="grid w-full grid-cols-2">
                    <TabsTrigger value="session" className="flex items-center gap-2">
                      <Database className="h-4 w-4" />
                      From Session
                    </TabsTrigger>
                    <TabsTrigger value="upload" className="flex items-center gap-2">
                      <Upload className="h-4 w-4" />
                      Upload File
                    </TabsTrigger>
                  </TabsList>

                  <TabsContent value="session" className="space-y-6 mt-6">
                    <div className="space-y-4">
                      <div>
                        <Label htmlFor="session-id">Session ID</Label>
                        <Input
                          id="session-id"
                          placeholder="Enter session ID from previous step..."
                          value={sessionId}
                          onChange={(e) => setSessionId(e.target.value)}
                        />
                      </div>

                      <div>
                        <Label htmlFor="target-variable">Target Variable *</Label>
                        <Input
                          id="target-variable"
                          placeholder="Enter the name of your target column (e.g., 'price', 'sales', 'category')"
                          value={targetVariable}
                          onChange={(e) => setTargetVariable(e.target.value)}
                        />
                      </div>

                      <div>
                        <Label htmlFor="instructions">Training Instructions</Label>
                        <Textarea
                          id="instructions"
                          placeholder="Describe your ML goals (e.g., 'Build a regression model to predict house prices with high accuracy')"
                          value={instructions}
                          onChange={(e) => setInstructions(e.target.value)}
                          rows={4}
                        />
                      </div>

                      <div className="flex justify-end">
                        <Button
                          onClick={handleTrainFromSession}
                          disabled={trainModelMutation.isPending}
                          className="bg-gradient-to-r from-indigo-500 to-blue-600 hover:from-indigo-600 hover:to-blue-700"
                        >
                          {trainModelMutation.isPending ? (
                            <>
                              <ProgressIndicator status="running" className="mr-2" />
                              Training Model...
                            </>
                          ) : (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" />
                              Train Model
                            </>
                          )}
                        </Button>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="upload" className="space-y-6 mt-6">
                    <div className="space-y-4">
                      <div>
                        <Label>Upload Dataset</Label>
                        <FileUploader
                          onUpload={handleFileUpload}
                          accept={['csv', 'xlsx', 'json']}
                          maxFiles={1}
                          className="mt-2"
                        />
                        {uploadedFiles.length > 0 && (
                          <div className="mt-2 text-sm text-green-600">
                            ✓ {uploadedFiles[0].file.name} uploaded
                          </div>
                        )}
                      </div>

                      <div>
                        <Label htmlFor="target-variable-upload">Target Variable *</Label>
                        <Input
                          id="target-variable-upload"
                          placeholder="Enter the name of your target column"
                          value={targetVariable}
                          onChange={(e) => setTargetVariable(e.target.value)}
                        />
                      </div>

                      <div>
                        <Label htmlFor="instructions-upload">Training Instructions</Label>
                        <Textarea
                          id="instructions-upload"
                          placeholder="Describe your ML goals..."
                          value={instructions}
                          onChange={(e) => setInstructions(e.target.value)}
                          rows={4}
                        />
                      </div>

                      <div className="flex justify-end">
                        <Button
                          onClick={handleTrainFromFile}
                          disabled={trainModelMutation.isPending || uploadedFiles.length === 0}
                          className="bg-gradient-to-r from-indigo-500 to-blue-600 hover:from-indigo-600 hover:to-blue-700"
                        >
                          {trainModelMutation.isPending ? (
                            <>
                              <ProgressIndicator status="running" className="mr-2" />
                              Training Model...
                            </>
                          ) : (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" />
                              Train Model
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
            {/* Training Options */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Training Options
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="runtime">Max Training Time (seconds)</Label>
                  <Input
                    id="runtime"
                    type="number"
                    value={trainingOptions.max_runtime_secs}
                    onChange={(e) => 
                      setTrainingOptions(prev => ({ ...prev, max_runtime_secs: parseInt(e.target.value) || 300 }))
                    }
                    min={60}
                    max={3600}
                  />
                  <p className="text-xs text-muted-foreground">
                    Longer training = better models (60s - 1 hour)
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="cv-folds">Cross-Validation Folds</Label>
                  <Input
                    id="cv-folds"
                    type="number"
                    value={trainingOptions.cv_folds}
                    onChange={(e) => 
                      setTrainingOptions(prev => ({ ...prev, cv_folds: parseInt(e.target.value) || 5 }))
                    }
                    min={3}
                    max={10}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="max-models">Max Models</Label>
                  <Input
                    id="max-models"
                    type="number"
                    value={trainingOptions.max_models}
                    onChange={(e) => 
                      setTrainingOptions(prev => ({ ...prev, max_models: parseInt(e.target.value) || 20 }))
                    }
                    min={5}
                    max={100}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="balance">Balance Classes</Label>
                  <Switch
                    id="balance"
                    checked={trainingOptions.balance_classes}
                    onCheckedChange={(checked) => 
                      setTrainingOptions(prev => ({ ...prev, balance_classes: checked }))
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="seed">Random Seed</Label>
                  <Input
                    id="seed"
                    type="number"
                    value={trainingOptions.seed}
                    onChange={(e) => 
                      setTrainingOptions(prev => ({ ...prev, seed: parseInt(e.target.value) || 42 }))
                    }
                  />
                </div>
              </CardContent>
            </Card>

            {/* Training Info */}
            <Card>
              <CardHeader>
                <CardTitle>H2O AutoML</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid gap-2">
                  {[
                    { algo: 'Random Forest', desc: 'Ensemble method' },
                    { algo: 'Gradient Boosting', desc: 'XGBoost, LightGBM' },
                    { algo: 'Deep Learning', desc: 'Neural networks' },
                    { algo: 'Linear Models', desc: 'GLM, ElasticNet' },
                    { algo: 'Naive Bayes', desc: 'Probabilistic' },
                    { algo: 'Ensemble', desc: 'Model stacking' },
                  ].map(algo => (
                    <div key={algo.algo} className="text-sm">
                      <div className="font-medium">{algo.algo}</div>
                      <div className="text-xs text-muted-foreground">{algo.desc}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Recent Sessions */}
            {recentSessions.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Recent Sessions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {recentSessions.map((session) => (
                      <div
                        key={session.sessionId}
                        className="flex items-center justify-between p-2 bg-gray-50 rounded cursor-pointer hover:bg-gray-100"
                        onClick={() => setSessionId(session.sessionId)}
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

            {/* Next Steps */}
            <Card>
              <CardHeader>
                <CardTitle>Workflow Chain</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">
                  After model training, you can:
                </p>
                <div className="space-y-2">
                  <Link href="/agents/prediction">
                    <Button variant="outline" size="sm" className="w-full justify-start">
                      <Target className="h-4 w-4 mr-2" />
                      Make Predictions
                    </Button>
                  </Link>
                  <Link href="/agents/visualization">
                    <Button variant="outline" size="sm" className="w-full justify-start">
                      <Database className="h-4 w-4 mr-2" />
                      Visualize Results
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>

            {/* Training Progress Info */}
            {trainModelMutation.isPending && (
              <Card className="border-indigo-200 bg-indigo-50">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-indigo-700">
                    <Clock className="h-5 w-5" />
                    Training in Progress
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <ProgressIndicator status="running" />
                    <p className="text-sm text-indigo-600">
                      H2O AutoML is training multiple models. This may take {Math.round((trainingOptions.max_runtime_secs || 300) / 60)} minutes.
                    </p>
                    <div className="text-xs text-indigo-500">
                      • Testing {trainingOptions.max_models} different algorithms<br/>
                      • Using {trainingOptions.cv_folds}-fold cross-validation<br/>
                      • Building ensemble models<br/>
                      • Ranking by performance
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}