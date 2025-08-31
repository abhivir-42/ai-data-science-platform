'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowLeft, Wrench, Database, Upload, PlayCircle, ArrowRight, Settings, Target } from 'lucide-react'
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
import { featureEngineeringClient } from '@/lib/uagent-client'
import { useSessionsStore } from '@/lib/store'
import { useToast } from '@/hooks/use-toast'
import type { EngineerFeaturesParams } from '@/lib/uagent-client'

interface FeatureOptions {
  create_polynomial?: boolean
  create_interactions?: boolean
  normalize_features?: boolean
  handle_categorical?: 'onehot' | 'label' | 'target'
  create_datetime_features?: boolean
}

interface UploadedFileData {
  file: File
  id: string
  status: 'pending' | 'uploading' | 'completed' | 'error'
  progress?: number
  base64?: string
  error?: string
}

export function FeatureEngineeringWorkspace() {
  const [activeTab, setActiveTab] = useState('session')
  const [sessionId, setSessionId] = useState('')
  const [instructions, setInstructions] = useState('')
  const [targetVariable, setTargetVariable] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFileData[]>([])
  const [featureOptions, setFeatureOptions] = useState<FeatureOptions>({
    create_polynomial: false,
    create_interactions: true,
    normalize_features: false,
    handle_categorical: 'onehot',
    create_datetime_features: true,
  })
  
  const router = useRouter()
  const { addSession, sessions } = useSessionsStore()
  const { toast } = useToast()

  // Feature engineering mutation
  const engineerFeaturesMutation = useMutation({
    mutationFn: async (params: EngineerFeaturesParams) => {
      return await featureEngineeringClient.engineerFeatures(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        // Add session to store
        addSession({
          sessionId: response.session_id,
          agentType: 'engineering',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: sessionId 
            ? `Engineered features from session ${sessionId.slice(0, 8)}...`
            : `Engineered features from uploaded data`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Features engineered successfully! 🔧",
          description: `Session ${response.session_id.slice(0, 8)}... created`,
          variant: "default",
        })

        // Navigate to session results
        router.push(`/sessions/${response.session_id}`)
      } else {
        throw new Error(response.error || 'Failed to engineer features')
      }
    },
    onError: (error) => {
      toast({
        title: "Feature engineering failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  const handleFileUpload = (files: UploadedFileData[]) => {
    setUploadedFiles(files)
  }

  const handleEngineerFromSession = () => {
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
        description: "Please specify the target variable for feature engineering (e.g., 'price', 'sales', 'category')",
        variant: "destructive",
      })
      return
    }

    engineerFeaturesMutation.mutate({
      session_id: sessionId,
      target_variable: targetVariable,
      user_instructions: instructions || undefined,
      feature_options: featureOptions,
    })
  }

  const handleEngineerFromFile = () => {
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
        description: "Please specify the target variable for feature engineering (e.g., 'price', 'sales', 'category')",
        variant: "destructive",
      })
      return
    }

    // Use the first file
    const file = uploadedFiles[0]
    engineerFeaturesMutation.mutate({
      filename: file.file.name,
      file_content: file.base64,
      target_variable: targetVariable,
      user_instructions: instructions || undefined,
      feature_options: featureOptions,
    })
  }

  // Get recent sessions for this agent
  const recentSessions = sessions
    .filter(s => s.agentType === 'engineering')
    .slice(0, 5)

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-amber-50 to-yellow-50">
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
                <div className="p-2 bg-orange-100 rounded-lg">
                  <Wrench className="h-6 w-6 text-orange-600" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">Feature Engineering</h1>
                  <p className="text-gray-600">Transform and create features to improve model performance</p>
                </div>
              </div>
            </div>

            {/* Main Interface */}
            <Card className="backdrop-blur-sm bg-white/80 border-white/20 shadow-xl">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Wrench className="h-5 w-5 text-orange-500" />
                  Feature Engineering Configuration
                </CardTitle>
                <CardDescription>
                  Create and transform features from your dataset to improve model performance
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
                          placeholder="Enter the name of your target column (e.g., 'price', 'sales')"
                          value={targetVariable}
                          onChange={(e) => setTargetVariable(e.target.value)}
                        />
                      </div>

                      <div>
                        <Label htmlFor="instructions">Feature Engineering Instructions</Label>
                        <Textarea
                          id="instructions"
                          placeholder="Describe what features you'd like to create (e.g., 'Create polynomial features for numerical columns and interaction terms between age and income')"
                          value={instructions}
                          onChange={(e) => setInstructions(e.target.value)}
                          rows={4}
                        />
                      </div>

                      <div className="flex justify-end">
                        <Button
                          onClick={handleEngineerFromSession}
                          disabled={engineerFeaturesMutation.isPending}
                          className="bg-gradient-to-r from-orange-500 to-amber-600 hover:from-orange-600 hover:to-amber-700"
                        >
                          {engineerFeaturesMutation.isPending ? (
                            <>
                              <ProgressIndicator status="running" className="mr-2" />
                              Engineering Features...
                            </>
                          ) : (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" />
                              Engineer Features
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
                        <Label htmlFor="instructions-upload">Feature Engineering Instructions</Label>
                        <Textarea
                          id="instructions-upload"
                          placeholder="Describe what features you'd like to create..."
                          value={instructions}
                          onChange={(e) => setInstructions(e.target.value)}
                          rows={4}
                        />
                      </div>

                      <div className="flex justify-end">
                        <Button
                          onClick={handleEngineerFromFile}
                          disabled={engineerFeaturesMutation.isPending || uploadedFiles.length === 0}
                          className="bg-gradient-to-r from-orange-500 to-amber-600 hover:from-orange-600 hover:to-amber-700"
                        >
                          {engineerFeaturesMutation.isPending ? (
                            <>
                              <ProgressIndicator status="running" className="mr-2" />
                              Engineering Features...
                            </>
                          ) : (
                            <>
                              <PlayCircle className="h-4 w-4 mr-2" />
                              Engineer Features
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
            {/* Feature Options */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Feature Options
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label htmlFor="polynomial">Polynomial Features</Label>
                  <Switch
                    id="polynomial"
                    checked={featureOptions.create_polynomial}
                    onCheckedChange={(checked) => 
                      setFeatureOptions(prev => ({ ...prev, create_polynomial: checked }))
                    }
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label htmlFor="interactions">Interaction Terms</Label>
                  <Switch
                    id="interactions"
                    checked={featureOptions.create_interactions}
                    onCheckedChange={(checked) => 
                      setFeatureOptions(prev => ({ ...prev, create_interactions: checked }))
                    }
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <Label htmlFor="normalize">Normalize Features</Label>
                  <Switch
                    id="normalize"
                    checked={featureOptions.normalize_features}
                    onCheckedChange={(checked) => 
                      setFeatureOptions(prev => ({ ...prev, normalize_features: checked }))
                    }
                  />
                </div>

                <div className="flex items-center justify-between">
                  <Label htmlFor="datetime">DateTime Features</Label>
                  <Switch
                    id="datetime"
                    checked={featureOptions.create_datetime_features}
                    onCheckedChange={(checked) => 
                      setFeatureOptions(prev => ({ ...prev, create_datetime_features: checked }))
                    }
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="categorical">Categorical Encoding</Label>
                  <Select
                    value={featureOptions.handle_categorical}
                    onValueChange={(value: 'onehot' | 'label' | 'target') => 
                      setFeatureOptions(prev => ({ ...prev, handle_categorical: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="onehot">One-Hot Encoding</SelectItem>
                      <SelectItem value="label">Label Encoding</SelectItem>
                      <SelectItem value="target">Target Encoding</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Feature Engineering Info */}
            <Card>
              <CardHeader>
                <CardTitle>Feature Types</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid gap-2">
                  {[
                    { type: 'Polynomial', desc: 'x², x³, etc.' },
                    { type: 'Interactions', desc: 'feature1 × feature2' },
                    { type: 'DateTime', desc: 'hour, day, month' },
                    { type: 'Categorical', desc: 'encode categories' },
                    { type: 'Statistical', desc: 'rolling averages' },
                    { type: 'Domain', desc: 'custom features' },
                  ].map(feature => (
                    <div key={feature.type} className="text-sm">
                      <div className="font-medium">{feature.type}</div>
                      <div className="text-xs text-muted-foreground">{feature.desc}</div>
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
                  After feature engineering, you can:
                </p>
                <div className="space-y-2">
                  <Link href="/agents/training">
                    <Button variant="outline" size="sm" className="w-full justify-start">
                      <Target className="h-4 w-4 mr-2" />
                      Train ML Model
                    </Button>
                  </Link>
                  <Link href="/agents/visualization">
                    <Button variant="outline" size="sm" className="w-full justify-start">
                      <Database className="h-4 w-4 mr-2" />
                      Visualize Features
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
