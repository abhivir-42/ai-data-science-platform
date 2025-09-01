'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowLeft, Plus, Play, ArrowRight, Settings, Trash2, Database, Sparkles, BarChart3, Wrench, Brain, Target, Zap, FileText, Upload } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { cn } from '@/lib/utils'
import { useWorkflowStore } from '@/lib/store'
import { useToast } from '@/hooks/use-toast'
import { workflowClient } from '@/lib/workflow-client'
import { WorkflowFileUploader } from './workflow-file-uploader'
import type { AgentType } from '@/lib/uagent-client'

// Agent configuration for workflow builder
const agentConfig = {
  loading: {
    title: 'Data Loader',
    description: 'Load files and datasets',
    icon: Database,
    gradient: 'bg-gradient-to-r from-blue-500 to-blue-600',
    bgColor: 'bg-blue-50 border-blue-200',
    textColor: 'text-blue-700',
  },
  cleaning: {
    title: 'Data Cleaning',
    description: 'Clean and preprocess data',
    icon: Sparkles,
    gradient: 'bg-gradient-to-r from-green-500 to-emerald-600',
    bgColor: 'bg-green-50 border-green-200',
    textColor: 'text-green-700',
  },
  visualization: {
    title: 'Data Visualization',
    description: 'Create charts and graphs',
    icon: BarChart3,
    gradient: 'bg-gradient-to-r from-purple-500 to-purple-600',
    bgColor: 'bg-purple-50 border-purple-200',
    textColor: 'text-purple-700',
  },
  engineering: {
    title: 'Feature Engineering',
    description: 'Transform and create features',
    icon: Wrench,
    gradient: 'bg-gradient-to-r from-orange-500 to-amber-600',
    bgColor: 'bg-orange-50 border-orange-200',
    textColor: 'text-orange-700',
  },
  training: {
    title: 'ML Training',
    description: 'Train machine learning models',
    icon: Brain,
    gradient: 'bg-gradient-to-r from-indigo-500 to-blue-600',
    bgColor: 'bg-indigo-50 border-indigo-200',
    textColor: 'text-indigo-700',
  },
  prediction: {
    title: 'ML Prediction',
    description: 'Make model predictions',
    icon: Target,
    gradient: 'bg-gradient-to-r from-red-500 to-pink-600',
    bgColor: 'bg-red-50 border-red-200',
    textColor: 'text-red-700',
  },
} as const

// Pre-defined workflow templates
const workflowTemplates = [
  {
    id: 'quick-analysis',
    name: 'Quick Data Analysis',
    description: 'Load → Clean → Visualize workflow for rapid insights',
    steps: ['loading', 'cleaning', 'visualization'] as AgentType[],
    icon: Zap,
    gradient: 'bg-gradient-to-r from-blue-600 to-purple-600',
  },
  {
    id: 'ml-pipeline',
    name: 'Complete ML Pipeline',
    description: 'End-to-end machine learning workflow',
    steps: ['loading', 'cleaning', 'engineering', 'training', 'prediction'] as AgentType[],
    icon: Brain,
    gradient: 'bg-gradient-to-r from-purple-600 to-indigo-600',
  },
  {
    id: 'data-prep',
    name: 'Data Preparation',
    description: 'Prepare and transform data for analysis',
    steps: ['loading', 'cleaning', 'engineering'] as AgentType[],
    icon: FileText,
    gradient: 'bg-gradient-to-r from-green-600 to-teal-600',
  },
]

export function WorkflowBuilder() {
  const [workflowName, setWorkflowName] = useState('')
  const [selectedSteps, setSelectedSteps] = useState<AgentType[]>([])
  const [showTemplates, setShowTemplates] = useState(true)
  const [showExecuteDialog, setShowExecuteDialog] = useState(false)
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isExecuting, setIsExecuting] = useState(false)
  const [executionResults, setExecutionResults] = useState<any>(null)
  
  const { workflows, createWorkflow, setActiveWorkflow } = useWorkflowStore()
  const { toast } = useToast()

  // Check for URL parameters to auto-start workflows
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const urlParams = new URLSearchParams(window.location.search)
      const template = urlParams.get('template')
      
      if (template === 'quick-analysis') {
        // Auto-open Quick Analysis execution dialog
        setSelectedTemplate('quick-analysis')
        setShowExecuteDialog(true)
        setShowTemplates(false)
        
        toast({
          title: "Quick Analysis Ready",
          description: "Upload your data file to start the Quick Data Analysis workflow",
          variant: "default",
        })
      }
    }
  }, [toast])

  const addStep = (agentType: AgentType) => {
    if (!selectedSteps.includes(agentType)) {
      setSelectedSteps([...selectedSteps, agentType])
    }
  }

  const removeStep = (agentType: AgentType) => {
    setSelectedSteps(selectedSteps.filter(step => step !== agentType))
  }

  const loadTemplate = (template: typeof workflowTemplates[0]) => {
    setSelectedTemplate(template.id)
    setShowExecuteDialog(true)
    setUploadedFile(null)
    setExecutionResults(null)
    
    toast({
      title: "Template selected",
      description: `Ready to execute "${template.name}" workflow`,
      variant: "default",
    })
  }

  const createNewWorkflow = () => {
    if (!workflowName.trim()) {
      toast({
        title: "Workflow name required",
        description: "Please enter a name for your workflow",
        variant: "destructive",
      })
      return
    }

    if (selectedSteps.length === 0) {
      toast({
        title: "No steps selected",
        description: "Please add at least one step to your workflow",
        variant: "destructive",
      })
      return
    }

    const workflow = createWorkflow(
      workflowName,
      selectedSteps.map(agentType => ({
        agentType,
        parameters: {},
      }))
    )

    toast({
      title: "Workflow created!",
      description: `"${workflowName}" workflow is ready to run`,
      variant: "default",
    })

    // Reset form
    setWorkflowName('')
    setSelectedSteps([])
    setShowTemplates(true)
  }

  const executeWorkflow = async () => {
    if (!uploadedFile) {
      toast({
        title: "File required",
        description: "Please upload a file to execute the workflow",
        variant: "destructive",
      })
      return
    }

    if (!selectedTemplate) {
      toast({
        title: "Template required", 
        description: "Please select a workflow template",
        variant: "destructive",
      })
      return
    }

    setIsExecuting(true)
    
    try {
      let result
      
      if (selectedTemplate === 'quick-analysis') {
        result = await workflowClient.executeQuickAnalysis(
          uploadedFile,
          'Perform comprehensive data analysis including loading, cleaning, and visualization'
        )
      } else if (selectedTemplate === 'ml-pipeline') {
        // For ML pipeline, we'd need to ask for target variable
        // For now, let's use a default message
        toast({
          title: "ML Pipeline",
          description: "ML Pipeline workflow requires target variable selection. This will be implemented in the next update.",
          variant: "default",
        })
        setIsExecuting(false)
        return
      } else {
        toast({
          title: "Template not supported",
          description: "This template is not yet supported for direct execution",
          variant: "destructive",
        })
        setIsExecuting(false)
        return
      }

      // Poll for results
      const finalResults = await workflowClient.pollWorkflowUntilComplete(
        result.workflow_id,
        (status) => {
          toast({
            title: "Workflow Progress",
            description: `Step ${status.current_step}/${status.total_steps}: ${Math.round(status.progress_percentage)}% complete`,
            variant: "default",
          })
        }
      )

      setExecutionResults(finalResults)
      
      toast({
        title: "Workflow completed!",
        description: `"${finalResults.name}" completed successfully in ${finalResults.total_execution_time?.toFixed(1)}s`,
        variant: "default",
      })

    } catch (error) {
      console.error('Workflow execution failed:', error)
      toast({
        title: "Workflow failed",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive",
      })
    } finally {
      setIsExecuting(false)
    }
  }

  const runWorkflow = (workflowId: string) => {
    const workflow = workflows.find(w => w.id === workflowId)
    if (workflow) {
      setActiveWorkflow(workflow)
      toast({
        title: "Custom workflows",
        description: "Custom workflow execution will be available in the next update",
        variant: "default",
      })
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Modern Header */}
      <div className="bg-white/80 backdrop-blur-xl border-b border-white/20 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Link href="/">
                <Button variant="ghost" size="sm" className="hover:bg-white/50">
                  <ArrowLeft className="mr-2 h-4 w-4" />
                  Back to Dashboard
                </Button>
              </Link>
              
              <div className="h-8 w-px bg-gray-300" />
              
              <div>
                <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                  Workflow Builder
                </h1>
                <p className="text-gray-600 mt-1">
                  Create and manage multi-step data science workflows
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Main Content */}
          <div className="lg:col-span-3 space-y-8">
            {/* Templates Section */}
            {showTemplates && (
              <Card className="border-0 bg-white/70 backdrop-blur-sm shadow-lg">
                <CardHeader>
                  <CardTitle className="text-2xl font-bold text-gray-900">Workflow Templates</CardTitle>
                  <CardDescription className="text-gray-600">
                    Start with a pre-built workflow template or create your own custom workflow
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {workflowTemplates.map((template) => {
                      const Icon = template.icon
                      return (
                        <div
                          key={template.id}
                          className="group relative overflow-hidden border border-gray-200 rounded-xl p-6 hover:shadow-xl transition-all duration-300 cursor-pointer bg-white hover:scale-105"
                          onClick={() => loadTemplate(template)}
                        >
                          <div className="absolute inset-0 bg-gradient-to-br from-white/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                          
                          <div className="relative z-10 space-y-4">
                            <div className="flex items-center justify-between">
                              <div className={cn('p-3 rounded-xl shadow-sm', template.gradient)}>
                                <Icon className="h-6 w-6 text-white" />
                              </div>
                              <Badge variant="outline" className="bg-white">
                                {template.steps.length} steps
                              </Badge>
                            </div>
                            
                            <div>
                              <h3 className="font-semibold text-gray-900 text-lg mb-2">{template.name}</h3>
                              <p className="text-sm text-gray-600 leading-relaxed">
                                {template.description}
                              </p>
                            </div>
                            
                            <div className="flex items-center flex-wrap gap-1">
                              {template.steps.map((stepType, index) => (
                                <div key={stepType} className="flex items-center">
                                  <div className={cn(
                                    'px-2 py-1 rounded text-xs font-medium',
                                    agentConfig[stepType].bgColor,
                                    agentConfig[stepType].textColor
                                  )}>
                                    {agentConfig[stepType].title}
                                  </div>
                                  {index < template.steps.length - 1 && (
                                    <ArrowRight className="h-3 w-3 mx-1 text-gray-400" />
                                  )}
                                </div>
                              ))}
                            </div>
                            
                            <Button className="w-full group-hover:bg-blue-600 transition-colors">
                              Use This Template
                            </Button>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                  
                  <div className="text-center pt-4 border-t border-gray-200">
                    <Button
                      variant="outline"
                      onClick={() => setShowTemplates(false)}
                      className="bg-white hover:bg-gray-50"
                    >
                      <Plus className="mr-2 h-4 w-4" />
                      Create Custom Workflow
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Custom Workflow Builder */}
            {!showTemplates && (
              <Card className="border-0 bg-white/70 backdrop-blur-sm shadow-lg">
                <CardHeader>
                  <CardTitle className="text-2xl font-bold text-gray-900">Build Your Workflow</CardTitle>
                  <CardDescription className="text-gray-600">
                    Configure your custom data science workflow step by step
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-8">
                  {/* Workflow Name */}
                  <div className="space-y-3">
                    <Label htmlFor="workflow-name" className="text-base font-medium text-gray-900">
                      Workflow Name
                    </Label>
                    <Input
                      id="workflow-name"
                      placeholder="Enter a descriptive name for your workflow..."
                      value={workflowName}
                      onChange={(e) => setWorkflowName(e.target.value)}
                      className="text-lg h-12 bg-white"
                    />
                  </div>

                  {/* Step Selection */}
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label className="text-base font-medium text-gray-900">Add Workflow Steps</Label>
                      <Select onValueChange={(value) => addStep(value as AgentType)}>
                        <SelectTrigger className="w-64 bg-white">
                          <SelectValue placeholder="Select a step to add..." />
                        </SelectTrigger>
                        <SelectContent>
                          {Object.entries(agentConfig).map(([agentType, config]) => {
                            const Icon = config.icon
                            return (
                              <SelectItem
                                key={agentType}
                                value={agentType}
                                disabled={selectedSteps.includes(agentType as AgentType)}
                                className="flex items-center py-3"
                              >
                                <div className="flex items-center space-x-3">
                                  <div className={cn('p-1 rounded', config.bgColor)}>
                                    <Icon className="h-4 w-4" />
                                  </div>
                                  <div>
                                    <div className="font-medium">{config.title}</div>
                                    <div className="text-xs text-gray-500">{config.description}</div>
                                  </div>
                                </div>
                              </SelectItem>
                            )
                          })}
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Selected Steps */}
                    {selectedSteps.length > 0 && (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <Label className="text-base font-medium text-gray-900">
                            Workflow Steps ({selectedSteps.length})
                          </Label>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setSelectedSteps([])}
                            className="text-gray-600"
                          >
                            Clear All
                          </Button>
                        </div>
                        
                        <div className="space-y-3">
                          {selectedSteps.map((stepType, index) => {
                            const config = agentConfig[stepType]
                            const Icon = config.icon
                            
                            return (
                              <div
                                key={`${stepType}-${index}`}
                                className="flex items-center space-x-4 p-4 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow"
                              >
                                <div className="flex items-center space-x-3">
                                  <div className="w-8 h-8 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-sm font-semibold">
                                    {index + 1}
                                  </div>
                                  <div className={cn('p-2 rounded-lg', config.bgColor)}>
                                    <Icon className="h-5 w-5" />
                                  </div>
                                </div>
                                
                                <div className="flex-1">
                                  <div className="flex items-center space-x-2 mb-1">
                                    <h3 className="font-semibold text-gray-900">{config.title}</h3>
                                    <Badge variant="outline" className={cn(config.bgColor, config.textColor)}>
                                      {stepType}
                                    </Badge>
                                  </div>
                                  <p className="text-sm text-gray-600">{config.description}</p>
                                </div>

                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => removeStep(stepType)}
                                  className="text-gray-400 hover:text-red-500"
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            )
                          })}
                        </div>
                        
                        {selectedSteps.length > 1 && (
                          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                            <div className="flex items-center text-sm text-blue-700">
                              <ArrowRight className="h-4 w-4 mr-2" />
                              <span className="font-medium">Data Flow:</span>
                              <div className="ml-2 flex items-center space-x-2">
                                {selectedSteps.map((stepType, index) => (
                                  <div key={stepType} className="flex items-center">
                                    <span>{agentConfig[stepType].title}</span>
                                    {index < selectedSteps.length - 1 && (
                                      <ArrowRight className="h-3 w-3 mx-2" />
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center justify-between pt-6 border-t border-gray-200">
                    <Button
                      variant="outline"
                      onClick={() => setShowTemplates(true)}
                      className="bg-white"
                    >
                      <ArrowLeft className="mr-2 h-4 w-4" />
                      Back to Templates
                    </Button>
                    
                    <Button
                      onClick={createNewWorkflow}
                      disabled={!workflowName.trim() || selectedSteps.length === 0}
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
                    >
                      <Plus className="mr-2 h-4 w-4" />
                      Create Workflow
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Existing Workflows */}
            {workflows.length > 0 && (
              <Card className="border-0 bg-white/70 backdrop-blur-sm shadow-lg">
                <CardHeader>
                  <CardTitle className="text-2xl font-bold text-gray-900">
                    Your Workflows ({workflows.length})
                  </CardTitle>
                  <CardDescription className="text-gray-600">
                    Manage and execute your created workflows
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {workflows.map((workflow) => (
                      <div
                        key={workflow.id}
                        className="flex items-center justify-between p-6 bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-shadow"
                      >
                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="font-semibold text-lg text-gray-900">{workflow.name}</h3>
                            <Badge variant={
                              workflow.status === 'completed' ? 'secondary' :
                              workflow.status === 'running' ? 'default' :
                              workflow.status === 'failed' ? 'destructive' : 'outline'
                            }>
                              {workflow.status}
                            </Badge>
                          </div>
                          
                          <div className="flex items-center space-x-2 mb-2">
                            {workflow.steps.map((step, index) => (
                              <div key={step.id} className="flex items-center">
                                <Badge variant="outline" className="text-xs">
                                  {agentConfig[step.agentType].title}
                                </Badge>
                                {index < workflow.steps.length - 1 && (
                                  <ArrowRight className="h-3 w-3 mx-1 text-gray-400" />
                                )}
                              </div>
                            ))}
                          </div>
                          
                          <p className="text-sm text-gray-500">
                            {workflow.steps.length} steps • Created {new Date(workflow.createdAt).toLocaleDateString()}
                          </p>
                        </div>

                        <Button
                          onClick={() => runWorkflow(workflow.id)}
                          className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                        >
                          <Play className="mr-2 h-4 w-4" />
                          Run Workflow
                        </Button>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="border-0 bg-white/70 backdrop-blur-sm shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-gray-900">Available Agents</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {Object.entries(agentConfig).map(([agentType, config]) => {
                  const Icon = config.icon
                  return (
                    <div key={agentType} className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <div className={cn('p-1 rounded', config.bgColor)}>
                          <Icon className="h-4 w-4" />
                        </div>
                        <div className="font-medium text-gray-900">{config.title}</div>
                      </div>
                      <p className="text-xs text-gray-600 ml-6">
                        {config.description}
                      </p>
                    </div>
                  )
                })}
              </CardContent>
            </Card>

            <Card className="border-0 bg-white/70 backdrop-blur-sm shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg font-semibold text-gray-900">Workflow Tips</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-gray-600">
                <div className="flex items-start space-x-2">
                  <div className="w-2 h-2 rounded-full bg-blue-500 mt-1.5" />
                  <p>Start with data loading or use existing sessions</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-2 h-2 rounded-full bg-green-500 mt-1.5" />
                  <p>Clean data before visualization or modeling</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-2 h-2 rounded-full bg-orange-500 mt-1.5" />
                  <p>Feature engineering improves ML performance</p>
                </div>
                <div className="flex items-start space-x-2">
                  <div className="w-2 h-2 rounded-full bg-purple-500 mt-1.5" />
                  <p>Save workflows as templates for reuse</p>
                </div>
              </CardContent>
            </Card>

            {workflows.length > 0 && (
              <Card className="border-0 bg-white/70 backdrop-blur-sm shadow-lg">
                <CardHeader>
                  <CardTitle className="text-lg font-semibold text-gray-900">Workflow Status</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3 text-sm">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Total Workflows:</span>
                      <span className="font-semibold text-gray-900">{workflows.length}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Completed:</span>
                      <span className="font-semibold text-green-600">
                        {workflows.filter(w => w.status === 'completed').length}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Running:</span>
                      <span className="font-semibold text-blue-600">
                        {workflows.filter(w => w.status === 'running').length}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>

      {/* Workflow Execution Dialog */}
      {showExecuteDialog && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-2xl font-bold text-gray-900">
                    Execute Workflow
                  </CardTitle>
                  <CardDescription className="text-gray-600">
                    {selectedTemplate === 'quick-analysis' && 'Quick Data Analysis: Load → Clean → Visualize'}
                    {selectedTemplate === 'ml-pipeline' && 'Complete ML Pipeline: Load → Clean → Engineer → Train'}
                    {selectedTemplate === 'data-prep' && 'Data Preparation: Load → Clean → Engineer'}
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowExecuteDialog(false)}
                  disabled={isExecuting}
                >
                  ×
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              {!executionResults ? (
                <>
                  <div>
                    <Label className="text-base font-medium text-gray-900 mb-4 block">
                      Upload Your Data File
                    </Label>
                    <WorkflowFileUploader
                      onFileSelect={setUploadedFile}
                      selectedFile={uploadedFile}
                      onRemoveFile={() => setUploadedFile(null)}
                    />
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-200">
                    <Button
                      variant="outline"
                      onClick={() => setShowExecuteDialog(false)}
                      disabled={isExecuting}
                    >
                      Cancel
                    </Button>
                    <Button
                      onClick={executeWorkflow}
                      disabled={!uploadedFile || isExecuting}
                      className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
                    >
                      {isExecuting ? (
                        <>
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin mr-2" />
                          Executing...
                        </>
                      ) : (
                        <>
                          <Play className="mr-2 h-4 w-4" />
                          Execute Workflow
                        </>
                      )}
                    </Button>
                  </div>
                </>
              ) : (
                <>
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex items-center text-green-800 mb-2">
                      <div className="w-5 h-5 bg-green-500 rounded-full flex items-center justify-center mr-2">
                        ✓
                      </div>
                      <span className="font-medium">Workflow Completed Successfully!</span>
                    </div>
                    <div className="text-sm text-green-700">
                      <div>Name: {executionResults.name}</div>
                      <div>Execution Time: {executionResults.total_execution_time?.toFixed(1)}s</div>
                      <div>Steps Completed: {executionResults.steps?.length}</div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="font-semibold text-gray-900">Workflow Results</h3>
                    {executionResults.steps?.map((step: any, index: number) => (
                      <div key={step.id} className="border border-gray-200 rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center space-x-2">
                            <div className="w-6 h-6 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center text-sm font-medium">
                              {index + 1}
                            </div>
                            <span className="font-medium capitalize">{step.agent_type} Agent</span>
                          </div>
                          <Badge variant={step.status === 'completed' ? 'secondary' : 'destructive'}>
                            {step.status}
                          </Badge>
                        </div>
                        {step.session_id && (
                          <div className="text-sm text-gray-600">
                            Session ID: {step.session_id}
                          </div>
                        )}
                        {step.execution_time_seconds && (
                          <div className="text-sm text-gray-600">
                            Execution Time: {step.execution_time_seconds.toFixed(1)}s
                          </div>
                        )}
                      </div>
                    ))}
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-gray-200">
                    <Button
                      variant="outline"
                      onClick={() => {
                        setShowExecuteDialog(false)
                        setExecutionResults(null)
                        setSelectedTemplate(null)
                        setUploadedFile(null)
                      }}
                    >
                      Close
                    </Button>
                    {executionResults.steps?.some((step: any) => step.session_id) && (
                      <Button
                        onClick={() => {
                          // Navigate to results view
                          const firstSessionId = executionResults.steps.find((step: any) => step.session_id)?.session_id
                          if (firstSessionId) {
                            window.location.href = `/sessions/${firstSessionId}`
                          }
                        }}
                        className="bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
                      >
                        View Results
                      </Button>
                    )}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}