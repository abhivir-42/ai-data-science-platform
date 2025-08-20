'use client'

import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { useRouter } from 'next/navigation'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { FileUploader } from '@/components/core/file-uploader'
import { ProgressIndicator } from '@/components/core/progress-indicator'
import { visualizationClient } from '@/lib/uagent-client'
import { useSessionsStore } from '@/lib/store'
import { useToast } from '@/hooks/use-toast'
import { BarChart3, ArrowRight } from 'lucide-react'

export function DataVisualizationWorkspace() {
  const [file, setFile] = useState<File | null>(null)
  const [instructions, setInstructions] = useState('')
  const router = useRouter()
  const { toast } = useToast()
  const { addSession } = useSessionsStore()

  const createChartMutation = useMutation({
    mutationFn: async (params: { file_content: string; filename?: string; user_instructions?: string }) => {
      return visualizationClient.createChart(params)
    },
    onSuccess: (response) => {
      if (response.success) {
        addSession({
          sessionId: response.session_id,
          agentType: 'visualization',
          createdAt: new Date().toISOString(),
          status: 'completed',
          description: `Created chart: ${file?.name || 'visualization'}`,
          executionTimeSeconds: response.execution_time_seconds,
        })

        toast({
          title: "Chart Creation Started",
          description: "Your visualization is being generated. Redirecting to results...",
        })
        router.push(`/sessions/${response.session_id}` as any)
      } else {
        throw new Error(response.error || 'Failed to create chart')
      }
    },
    onError: (error) => {
      toast({
        title: "Chart Creation Failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      })
    },
  })

  const handleCreateChart = async () => {
    if (!file) {
      toast({
        title: "No File Selected",
        description: "Please select a CSV file to visualize",
        variant: "destructive",
      })
      return
    }

    try {
      const base64 = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => {
          const result = reader.result as string
          const base64 = result.split(',')[1]
          resolve(base64)
        }
        reader.onerror = reject
        reader.readAsDataURL(file)
      })

      await createChartMutation.mutateAsync({
        file_content: base64,
        filename: file.name,
        user_instructions: instructions.trim() || undefined,
      })
    } catch (error) {
      toast({
        title: "File Processing Error",
        description: "Failed to process the selected file",
        variant: "destructive",
      })
    }
  }

  const isLoading = createChartMutation.isPending

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50/30 via-white to-purple-50/30 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r from-blue-500 to-purple-600 text-white mb-4">
            <BarChart3 className="h-8 w-8" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
            Data Visualization Workspace
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Upload your CSV data and let AI create stunning Plotly visualizations with intelligent chart recommendations
          </p>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Configuration Panel */}
          <Card className="backdrop-blur-sm bg-white/80 border-white/20 shadow-xl">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5 text-blue-500" />
                Chart Configuration
              </CardTitle>
              <CardDescription>
                Upload your data and specify visualization preferences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* File Upload */}
              <div>
                <Label htmlFor="file-upload">Data File</Label>
                <FileUploader
                  onUpload={(uploadedFiles) => {
                    if (uploadedFiles.length > 0) {
                      setFile(uploadedFiles[0].file)
                    }
                  }}
                  accept={['csv']}
                  maxFiles={1}
                  maxSize={10 * 1024 * 1024} // 10MB
                  className="mt-2"
                />
                {file && (
                  <p className="text-sm text-muted-foreground mt-2">
                    Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
                  </p>
                )}
              </div>

              {/* Instructions */}
              <div>
                <Label htmlFor="instructions">Visualization Instructions (Optional)</Label>
                <Textarea
                  id="instructions"
                  placeholder="e.g., Create a scatter plot of age vs salary with different colors for departments, add a trend line..."
                  value={instructions}
                  onChange={(e) => setInstructions(e.target.value)}
                  className="mt-2 min-h-[100px]"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Describe the type of chart you want and any specific formatting preferences
                </p>
              </div>

              {/* Action Button */}
              <Button
                onClick={handleCreateChart}
                disabled={!file || isLoading}
                className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <ProgressIndicator status="running" variant="compact" />
                    Creating Visualization...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" />
                    Create Chart
                    <ArrowRight className="h-4 w-4" />
                  </div>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Info Panel */}
          <Card className="backdrop-blur-sm bg-white/80 border-white/20 shadow-xl">
            <CardHeader>
              <CardTitle className="text-purple-600">AI-Powered Visualization</CardTitle>
              <CardDescription>
                Advanced chart generation with intelligent recommendations
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium">Smart Chart Selection</h3>
                    <p className="text-sm text-muted-foreground">
                      AI analyzes your data structure and recommends optimal chart types
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-purple-500 mt-2 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium">Interactive Plotly Charts</h3>
                    <p className="text-sm text-muted-foreground">
                      Generate responsive, interactive visualizations with professional styling
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-green-500 mt-2 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium">Code Generation</h3>
                    <p className="text-sm text-muted-foreground">
                      Get the Python code to recreate your visualizations
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-2 h-2 rounded-full bg-orange-500 mt-2 flex-shrink-0" />
                  <div>
                    <h3 className="font-medium">Custom Instructions</h3>
                    <p className="text-sm text-muted-foreground">
                      Guide the AI with specific chart preferences and styling requests
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Tips */}
        <Card className="backdrop-blur-sm bg-gradient-to-r from-blue-50/50 to-purple-50/50 border-blue-200/50">
          <CardContent className="pt-6">
            <h3 className="font-semibold text-blue-700 mb-3">💡 Quick Tips</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                • <strong>Chart Types:</strong> Ask for scatter plots, bar charts, line graphs, histograms, box plots
              </div>
              <div>
                • <strong>Customization:</strong> Specify colors, titles, axis labels, and themes
              </div>
              <div>
                • <strong>Data Insights:</strong> Request trend lines, correlations, or statistical overlays
              </div>
              <div>
                • <strong>Best Results:</strong> Clean CSV files with clear column headers work best
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
