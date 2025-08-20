'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { ArrowLeft, Upload, Folder, FileText, PlayCircle, ArrowRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { FileUploader } from '@/components/core/file-uploader'
import { ProgressIndicator } from '@/components/core/progress-indicator'
import { dataLoaderClient } from '@/lib/uagent-client'
import { useSessionsStore } from '@/lib/store'
import { useToast } from '@/hooks/use-toast'
import type { LoadFileParams, LoadDirectoryParams } from '@/lib/uagent-client'

export function DataLoaderWorkspace() {
  const [activeTab, setActiveTab] = useState('file')
  const [directoryPath, setDirectoryPath] = useState('')
  const [instructions, setInstructions] = useState('')
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([])
  
  const router = useRouter()
  const { addSession } = useSessionsStore()
  const { toast } = useToast()

  // File loading mutation
  const loadFileMutation = useMutation({
    mutationFn: async (params: LoadFileParams) => {
      return await dataLoaderClient.loadFile(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        // Add session to store
        addSession({
          sessionId: response.session_id,
          agentType: 'loading',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: `Loaded ${uploadedFiles.length} file(s)`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Files loaded successfully!",
          description: `Session ${response.session_id.slice(0, 8)}... created`,
          variant: "default",
        })

        // Navigate to session results
        router.push(`/sessions/${response.session_id}` as any)
      } else {
        throw new Error(response.error || 'Failed to load files')
      }
    },
    onError: (error) => {
      toast({
        title: "File loading failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  // Directory loading mutation
  const loadDirectoryMutation = useMutation({
    mutationFn: async (params: LoadDirectoryParams) => {
      return await dataLoaderClient.loadDirectory(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        addSession({
          sessionId: response.session_id,
          agentType: 'loading',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: `Loaded directory: ${directoryPath}`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Directory loaded successfully!",
          description: `Session ${response.session_id.slice(0, 8)}... created`,
          variant: "default",
        })

        router.push(`/sessions/${response.session_id}` as any)
      } else {
        throw new Error(response.error || 'Failed to load directory')
      }
    },
    onError: (error) => {
      toast({
        title: "Directory loading failed",
        description: error instanceof Error ? error.message : 'Unknown error occurred',
        variant: "destructive",
      })
    }
  })

  const handleFileUpload = (files: any[]) => {
    setUploadedFiles(files)
  }

  const handleLoadFiles = () => {
    if (uploadedFiles.length === 0) {
      toast({
        title: "No files selected",
        description: "Please upload files before proceeding",
        variant: "destructive",
      })
      return
    }

    // Use the first file for now (extend later for multiple files)
    const file = uploadedFiles[0]
    loadFileMutation.mutate({
      filename: file.file.name,
      file_content: file.base64!,
      user_instructions: instructions || undefined,
    })
  }

  const handleLoadDirectory = () => {
    if (!directoryPath.trim()) {
      toast({
        title: "Directory path required",
        description: "Please enter a directory path",
        variant: "destructive",
      })
      return
    }

    loadDirectoryMutation.mutate({
      directory_path: directoryPath,
      user_instructions: instructions || undefined,
    })
  }

  const isLoading = loadFileMutation.isPending || loadDirectoryMutation.isPending

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
                <h1 className="text-2xl font-bold">Data Loader</h1>
                <p className="text-muted-foreground">
                  Upload and process datasets from files or directories
                </p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2">
              <div className="text-right text-xs text-muted-foreground">
                <div>Port: 8005</div>
                <div>Agent: DataLoaderToolsAgent</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-4 py-8">
        {isLoading && (
          <div className="mb-6">
            <ProgressIndicator
              status="running"
              message={
                loadFileMutation.isPending 
                  ? "Loading and processing files..."
                  : "Loading directory contents..."
              }
              variant="detailed"
            />
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="file" className="flex items-center space-x-2">
                  <Upload className="h-4 w-4" />
                  <span>Upload Files</span>
                </TabsTrigger>
                <TabsTrigger value="directory" className="flex items-center space-x-2">
                  <Folder className="h-4 w-4" />
                  <span>Load Directory</span>
                </TabsTrigger>
                <TabsTrigger value="pdf" className="flex items-center space-x-2">
                  <FileText className="h-4 w-4" />
                  <span>Extract PDF</span>
                </TabsTrigger>
              </TabsList>

              {/* File Upload Tab */}
              <TabsContent value="file" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Upload className="h-5 w-5" />
                      <span>Upload Dataset Files</span>
                    </CardTitle>
                    <CardDescription>
                      Upload CSV, Excel, JSON, Parquet, or TSV files for analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <FileUploader
                      accept={['csv', 'xlsx', 'xls', 'json', 'parquet', 'tsv']}
                      multiple={false} // For now, handle one file at a time
                      onUpload={handleFileUpload}
                      title="Dataset Upload"
                      description="Select your data file to begin processing"
                    />
                  </CardContent>
                </Card>

                {uploadedFiles.length > 0 && (
                  <Card>
                    <CardHeader>
                      <CardTitle>Processing Options</CardTitle>
                      <CardDescription>
                        Optional instructions for data processing
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="instructions">Processing Instructions (Optional)</Label>
                        <Textarea
                          id="instructions"
                          placeholder="e.g., Parse dates in column 'timestamp', treat empty strings as null..."
                          value={instructions}
                          onChange={(e) => setInstructions(e.target.value)}
                          rows={3}
                        />
                      </div>

                      <Button 
                        onClick={handleLoadFiles} 
                        disabled={isLoading}
                        className="w-full"
                        size="lg"
                      >
                        {isLoading ? (
                          <>Processing Files...</>
                        ) : (
                          <>
                            <PlayCircle className="mr-2 h-5 w-5" />
                            Load and Process Files
                          </>
                        )}
                      </Button>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              {/* Directory Tab */}
              <TabsContent value="directory" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Folder className="h-5 w-5" />
                      <span>Load Directory</span>
                    </CardTitle>
                    <CardDescription>
                      Process all data files within a directory
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="directory">Directory Path</Label>
                      <Input
                        id="directory"
                        type="text"
                        placeholder="/path/to/your/data/directory"
                        value={directoryPath}
                        onChange={(e) => setDirectoryPath(e.target.value)}
                      />
                      <p className="text-sm text-muted-foreground">
                        Enter the full path to the directory containing your data files
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="dir-instructions">Processing Instructions (Optional)</Label>
                      <Textarea
                        id="dir-instructions"
                        placeholder="e.g., Only process CSV files, ignore hidden files..."
                        value={instructions}
                        onChange={(e) => setInstructions(e.target.value)}
                        rows={3}
                      />
                    </div>

                    <Button 
                      onClick={handleLoadDirectory} 
                      disabled={isLoading || !directoryPath.trim()}
                      className="w-full"
                      size="lg"
                    >
                      {isLoading ? (
                        <>Processing Directory...</>
                      ) : (
                        <>
                          <PlayCircle className="mr-2 h-5 w-5" />
                          Load Directory
                        </>
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* PDF Tab */}
              <TabsContent value="pdf" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <FileText className="h-5 w-5" />
                      <span>Extract PDF Data</span>
                    </CardTitle>
                    <CardDescription>
                      Extract tables and text from PDF documents
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="text-center py-12 text-muted-foreground">
                      <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                      <p className="text-lg font-medium mb-2">PDF Extraction</p>
                      <p>This feature will be available soon</p>
                      <p className="text-sm mt-4">
                        Use the File Upload tab to upload other data formats
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Supported Formats</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { ext: 'CSV', desc: 'Comma-separated values' },
                    { ext: 'Excel', desc: 'XLSX/XLS files' },
                    { ext: 'JSON', desc: 'JavaScript Object Notation' },
                    { ext: 'Parquet', desc: 'Columnar storage format' },
                    { ext: 'TSV', desc: 'Tab-separated values' },
                    { ext: 'PDF', desc: 'Coming soon' },
                  ].map(format => (
                    <div key={format.ext} className="text-sm">
                      <div className="font-medium">{format.ext}</div>
                      <div className="text-xs text-muted-foreground">{format.desc}</div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Next Steps</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-6 h-6 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-xs">
                    1
                  </div>
                  <span>Upload your data files</span>
                </div>
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-6 h-6 rounded-full bg-muted text-muted-foreground flex items-center justify-center text-xs">
                    2
                  </div>
                  <span>Review loaded data</span>
                </div>
                <div className="flex items-center space-x-2 text-sm">
                  <div className="w-6 h-6 rounded-full bg-muted text-muted-foreground flex items-center justify-center text-xs">
                    3
                  </div>
                  <span>Clean and preprocess</span>
                </div>
                
                <Link href="/agents/cleaning" className="block mt-4">
                  <Button variant="outline" className="w-full">
                    Data Cleaning
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Tips</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p>• Ensure your files have clear column headers</p>
                <p>• Check for consistent data types in columns</p>
                <p>• Large files (&gt;100MB) may take longer to process</p>
                <p>• Directory loading processes all compatible files</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
