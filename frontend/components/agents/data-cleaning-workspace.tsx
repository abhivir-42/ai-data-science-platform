'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowLeft, Sparkles, Database, Upload, PlayCircle, ArrowRight, Settings } from 'lucide-react'
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
import { dataCleaningClient } from '@/lib/uagent-client'
import { useSessionsStore } from '@/lib/store'
import { useToast } from '@/hooks/use-toast'
import type { CleanDataParams } from '@/lib/uagent-client'

interface AdvancedOptions {
  remove_duplicates?: boolean
  handle_missing?: 'drop' | 'fill' | 'interpolate'
  normalize_columns?: boolean
  detect_outliers?: boolean
}

export function DataCleaningWorkspace() {
  const [activeTab, setActiveTab] = useState('session')
  const [sessionId, setSessionId] = useState('')
  const [instructions, setInstructions] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  const [advancedOptions, setAdvancedOptions] = useState<AdvancedOptions>({
    remove_duplicates: true,
    handle_missing: 'fill',
    normalize_columns: false,
    detect_outliers: true,
  })
  
  const router = useRouter()
  const { addSession, sessions } = useSessionsStore()
  const { toast } = useToast()

  // Data cleaning mutation
  const cleanDataMutation = useMutation({
    mutationFn: async (params: CleanDataParams) => {
      return await dataCleaningClient.cleanData(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        // Add session to store
        addSession({
          sessionId: response.session_id,
          agentType: 'cleaning',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: sessionId 
            ? `Cleaned data from session ${sessionId.slice(0, 8)}...`
            : `Cleaned uploaded data`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Data cleaned successfully!",
          description: `Session ${response.session_id.slice(0, 8)}... created`,
          variant: "default",
        })

        // Navigate to session results
        router.push(`/sessions/${response.session_id}` as any)
      } else {
        throw new Error(response.error || 'Failed to clean data')
      }
    },
    onError: (error) => {
      toast({
        title: "Data cleaning failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  const handleFileUpload = (files: any[]) => {
    setUploadedFiles(files)
  }

  const handleCleanFromSession = () => {
    if (!sessionId.trim()) {
      toast({
        title: "Session ID required",
        description: "Please enter a session ID",
        variant: "destructive",
      })
      return
    }

    cleanDataMutation.mutate({
      session_id: sessionId,
      user_instructions: instructions || undefined,
      advanced_options: advancedOptions,
    })
  }

  const handleCleanFromFile = () => {
    if (uploadedFiles.length === 0) {
      toast({
        title: "No files selected",
        description: "Please upload a file before proceeding",
        variant: "destructive",
      })
      return
    }

    // Use the first file
    const file = uploadedFiles[0]
    cleanDataMutation.mutate({
      filename: file.file.name,
      file_content: file.base64!,
      user_instructions: instructions || undefined,
      advanced_options: advancedOptions,
    })
  }

  const updateAdvancedOption = (key: keyof AdvancedOptions, value: any) => {
    setAdvancedOptions(prev => ({ ...prev, [key]: value }))
  }

  const isLoading = cleanDataMutation.isPending

  // Get recent data loader sessions for quick reference
  const dataLoaderSessions = sessions
    .filter(s => s.agentType === 'loading' && s.status === 'completed')
    .slice(0, 5)

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
                <h1 className="text-2xl font-bold flex items-center space-x-2">
                  <Sparkles className="h-6 w-6 text-green-600" />
                  <span>Data Cleaning</span>
                </h1>
                <p className="text-muted-foreground">
                  Clean and preprocess your data with AI-powered recommendations
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="text-right text-xs text-muted-foreground">
                <div>Port: 8004</div>
                <div>Agent: DataCleaningAgent</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="session" className="flex items-center space-x-2">
                  <Database className="h-4 w-4" />
                  <span>From Session</span>
                </TabsTrigger>
                <TabsTrigger value="upload" className="flex items-center space-x-2">
                  <Upload className="h-4 w-4" />
                  <span>Upload File</span>
                </TabsTrigger>
              </TabsList>

              {/* From Session Tab */}
              <TabsContent value="session" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Database className="h-5 w-5" />
                      <span>Clean Data from Previous Session</span>
                    </CardTitle>
                    <CardDescription>
                      Use data from a previous data loading session
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="session-id">Session ID</Label>
                      <Input
                        id="session-id"
                        type="text"
                        placeholder="Enter session ID from data loader..."
                        value={sessionId}
                        onChange={(e) => setSessionId(e.target.value)}
                      />
                      <p className="text-sm text-muted-foreground">
                        Use a session ID from the Data Loader agent
                      </p>
                    </div>

                    {dataLoaderSessions.length > 0 && (
                      <div className="space-y-2">
                        <Label>Recent Data Loader Sessions</Label>
                        <div className="grid gap-2">
                          {dataLoaderSessions.map((session) => (
                            <div
                              key={session.sessionId}
                              className="flex items-center justify-between p-2 border rounded cursor-pointer hover:bg-muted/50"
                              onClick={() => setSessionId(session.sessionId)}
                            >
                              <div className="flex-1">
                                <div className="text-sm font-medium">
                                  {session.description || `Session ${session.sessionId.slice(0, 8)}...`}
                                </div>
                                <div className="text-xs text-muted-foreground">
                                  {new Date(session.createdAt).toLocaleString()}
                                </div>
                              </div>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setSessionId(session.sessionId)
                                }}
                              >
                                Use
                              </Button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Upload File Tab */}
              <TabsContent value="upload" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Upload className="h-5 w-5" />
                      <span>Upload CSV File</span>
                    </CardTitle>
                    <CardDescription>
                      Upload a CSV file directly for cleaning
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <FileUploader
                      accept={['csv']}
                      multiple={false}
                      onUpload={handleFileUpload}
                      title="CSV File Upload"
                      description="Select a CSV file to clean and preprocess"
                    />
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            {/* Cleaning Instructions */}
            <Card>
              <CardHeader>
                <CardTitle>Cleaning Instructions</CardTitle>
                <CardDescription>
                  Provide specific instructions for how to clean your data
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="instructions">Custom Instructions (Optional)</Label>
                  <Textarea
                    id="instructions"
                    placeholder="e.g., Remove outliers in the 'price' column, fill missing values in 'category' with 'unknown', convert all text to lowercase..."
                    value={instructions}
                    onChange={(e) => setInstructions(e.target.value)}
                    rows={4}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Advanced Options */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Settings className="h-5 w-5" />
                  <span>Advanced Cleaning Options</span>
                </CardTitle>
                <CardDescription>
                  Configure automated cleaning operations
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Remove Duplicates</Label>
                      <p className="text-sm text-muted-foreground">
                        Automatically remove duplicate rows
                      </p>
                    </div>
                    <Switch
                      checked={advancedOptions.remove_duplicates}
                      onCheckedChange={(checked: boolean) => updateAdvancedOption('remove_duplicates', checked)}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Handle Missing Values</Label>
                    <Select
                      value={advancedOptions.handle_missing}
                      onValueChange={(value) => updateAdvancedOption('handle_missing', value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="drop">Drop rows/columns</SelectItem>
                        <SelectItem value="fill">Fill with defaults</SelectItem>
                        <SelectItem value="interpolate">Interpolate values</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Normalize Columns</Label>
                      <p className="text-sm text-muted-foreground">
                        Standardize column names and formats
                      </p>
                    </div>
                    <Switch
                      checked={advancedOptions.normalize_columns}
                      onCheckedChange={(checked: boolean) => updateAdvancedOption('normalize_columns', checked)}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Detect Outliers</Label>
                      <p className="text-sm text-muted-foreground">
                        Identify and flag potential outliers
                      </p>
                    </div>
                    <Switch
                      checked={advancedOptions.detect_outliers}
                      onCheckedChange={(checked: boolean) => updateAdvancedOption('detect_outliers', checked)}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Execute Button */}
            <Card>
              <CardContent className="pt-6 space-y-4">
                {/* 🔥 PROGRESS INDICATOR - RIGHT WHERE USER EXPECTS IT */}
                {isLoading && (
                  <ProgressIndicator
                    status="running"
                    message="Analyzing and cleaning your data..."
                    variant="detailed"
                  />
                )}
                
                <Button 
                  onClick={activeTab === 'session' ? handleCleanFromSession : handleCleanFromFile}
                  disabled={
                    isLoading || 
                    (activeTab === 'session' && !sessionId.trim()) ||
                    (activeTab === 'upload' && uploadedFiles.length === 0)
                  }
                  className="w-full"
                  size="lg"
                >
                  {isLoading ? (
                    <>Cleaning Data...</>
                  ) : (
                    <>
                      <PlayCircle className="mr-2 h-5 w-5" />
                      Clean Data
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Cleaning Operations</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-3 text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>Missing value detection</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>Duplicate row removal</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>Data type validation</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>Outlier detection</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>Column normalization</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 rounded-full bg-green-500"></div>
                    <span>Format standardization</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Workflow Progress</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-xs">
                    ✓
                  </div>
                  <span>Data loaded</span>
                </div>
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-6 h-6 rounded-full bg-orange-500 text-white flex items-center justify-center text-xs">
                    🧹
                  </div>
                  <span>Clean your data</span>
                </div>
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-6 h-6 rounded-full bg-muted text-muted-foreground flex items-center justify-center text-xs">
                    3
                  </div>
                  <span>Next: Analysis & insights</span>
                </div>
                
                <div className="mt-4 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                  <p className="text-sm text-blue-700 font-medium">
                    💡 After cleaning, visit the Results page to see your next steps and continue the workflow!
                  </p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Cleaning Tips</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p>• Review data quality before cleaning</p>
                <p>• Start with basic operations first</p>
                <p>• Use custom instructions for domain-specific cleaning</p>
                <p>• Check the generated code for transparency</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
