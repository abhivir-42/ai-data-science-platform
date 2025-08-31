'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { ArrowRight, Database, Sparkles, BarChart3, Wrench, Brain, Target, Clock, Play, Users, FileText, Zap } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { cn, formatRelativeTime } from '@/lib/utils'
import { useSessionsStore } from '@/lib/store'
import type { AgentType } from '@/lib/uagent-client'

// Agent configuration with proper styling
const agentConfig: Record<AgentType, {
  title: string
  description: string
  icon: any
  gradient: string
  iconBg: string
  port: number
  capabilities: string[]
  isReady: boolean
}> = {
  loading: {
    title: 'Data Loader',
    description: 'Upload and process CSV, Excel, JSON, PDF, and Parquet files with intelligent schema detection',
    icon: Database,
    gradient: 'bg-gradient-to-br from-blue-500 to-blue-600',
    iconBg: 'bg-blue-100 text-blue-600',
    port: 8005,
    capabilities: ['File Upload', 'Format Detection', 'Schema Inference', 'Data Validation'],
    isReady: true,
  },
  cleaning: {
    title: 'Data Cleaning',
    description: 'Clean and preprocess your data with AI-powered recommendations and automated transformations',
    icon: Sparkles,
    gradient: 'bg-gradient-to-br from-green-500 to-emerald-600',
    iconBg: 'bg-green-100 text-green-600',
    port: 8004,
    capabilities: ['Missing Values', 'Duplicates', 'Outliers', 'Normalization'],
    isReady: true,
  },
  visualization: {
    title: 'Data Visualization',
    description: 'Create interactive charts and visualizations with Plotly and AI-guided insights',
    icon: BarChart3,
    gradient: 'bg-gradient-to-br from-purple-500 to-purple-600',
    iconBg: 'bg-purple-100 text-purple-600',
    port: 8006,
    capabilities: ['Charts', 'Plots', 'Interactive Viz', 'Export Options'],
    isReady: true,
  },
  engineering: {
    title: 'Feature Engineering',
    description: 'Engineer and transform features for machine learning with automated feature selection',
    icon: Wrench,
    gradient: 'bg-gradient-to-br from-orange-500 to-amber-600',
    iconBg: 'bg-orange-100 text-orange-600',
    port: 8007,
    capabilities: ['Feature Creation', 'Transformations', 'Encoding', 'Scaling'],
    isReady: true,
  },
  training: {
    title: 'ML Training',
    description: 'Train machine learning models with H2O AutoML and automated hyperparameter tuning',
    icon: Brain,
    gradient: 'bg-gradient-to-br from-indigo-500 to-blue-600',
    iconBg: 'bg-indigo-100 text-indigo-600',
    port: 8008,
    capabilities: ['AutoML', 'Model Selection', 'Hyperparameter Tuning', 'Leaderboard'],
    isReady: true,
  },
  prediction: {
    title: 'ML Prediction',
    description: 'Make predictions and analyze model performance with comprehensive evaluation metrics',
    icon: Target,
    gradient: 'bg-gradient-to-br from-red-500 to-pink-600',
    iconBg: 'bg-red-100 text-red-600',
    port: 8009,
    capabilities: ['Single Prediction', 'Batch Prediction', 'Model Analysis', 'Explanations'],
    isReady: true,
  },
}

// Workflow templates with modern design
const workflowTemplates = [
  {
    id: 'quick-analysis',
    title: 'Quick Data Analysis',
    description: 'Load → Clean → Visualize your data in minutes',
    steps: ['loading', 'cleaning', 'visualization'] as AgentType[],
    icon: Zap,
    estimatedTime: '5-10 min',
    difficulty: 'Beginner',
    gradient: 'bg-gradient-to-r from-blue-600 via-green-500 to-purple-600',
  },
  {
    id: 'ml-pipeline',
    title: 'Complete ML Pipeline',
    description: 'End-to-end machine learning workflow from data to predictions',
    steps: ['loading', 'cleaning', 'engineering', 'training', 'prediction'] as AgentType[],
    icon: Brain,
    estimatedTime: '20-30 min',
    difficulty: 'Advanced',
    gradient: 'bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600',
  },
  {
    id: 'data-prep',
    title: 'Data Preparation',
    description: 'Clean and prepare your data for analysis with feature engineering',
    steps: ['loading', 'cleaning', 'engineering'] as AgentType[],
    icon: FileText,
    estimatedTime: '10-15 min',
    difficulty: 'Intermediate', 
    gradient: 'bg-gradient-to-r from-green-600 via-teal-500 to-blue-500',
  },
]

export function WorkflowDashboard() {
  const { sessions, getSessionsByAgent } = useSessionsStore()
  const [agentStats, setAgentStats] = useState<Record<AgentType, number>>({} as any)
  const [isHydrated, setIsHydrated] = useState(false)

  // Hydration guard
  useEffect(() => {
    setIsHydrated(true)
  }, [])

  // Calculate agent usage stats
  useEffect(() => {
    const stats = {} as Record<AgentType, number>
    Object.keys(agentConfig).forEach(agentType => {
      stats[agentType as AgentType] = getSessionsByAgent(agentType as AgentType).length
    })
    setAgentStats(stats)
  }, [sessions, getSessionsByAgent])

  const recentSessions = sessions.slice(0, 5)

  // Don't render until hydrated to prevent hydration mismatch
  if (!isHydrated) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2 mb-8"></div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-64 bg-gray-200 rounded"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Modern Header with Glass Effect */}
      <div className="bg-white/80 backdrop-blur-xl border-b border-white/20 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
            <div className="mb-4 sm:mb-0">
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                AI Data Science Platform
              </h1>
              <p className="mt-2 text-xl text-gray-600">
                Transform your data into insights with workflow-driven AI agents
              </p>
            </div>
            
            <div className="flex items-center space-x-6">
              <div className="text-right">
                <div className="text-2xl font-bold text-gray-900">{sessions.length}</div>
                <div className="text-sm text-gray-500">Total Sessions</div>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-blue-600">{Object.values(agentStats).reduce((a, b) => a + b, 0)}</div>
                <div className="text-sm text-gray-500">Executions</div>
              </div>
              <Link href="/workflows">
                <Button className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-lg">
                  <Users className="mr-2 h-4 w-4" />
                  Workflow Builder
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-12">
        {/* Quick Start Workflows */}
        <section>
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Quick Start Workflows</h2>
            <p className="text-lg text-gray-600">Choose a pre-built workflow to get started instantly</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {workflowTemplates.map((template) => {
              const Icon = template.icon
              return (
                <Card key={template.id} className="group relative overflow-hidden hover:shadow-xl transition-all duration-300 border-0 bg-white/70 backdrop-blur-sm">
                  <div className="absolute inset-0 bg-gradient-to-br from-white/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                  
                  <CardHeader className="relative z-10">
                    <div className="flex items-start justify-between mb-4">
                      <div className={cn('p-3 rounded-xl shadow-sm', template.gradient)}>
                        <Icon className="h-6 w-6 text-white" />
                      </div>
                      <div className="flex flex-col items-end space-y-2">
                        <Badge variant="secondary" className="font-medium">
                          {template.difficulty}
                        </Badge>
                        <div className="flex items-center text-sm text-gray-500">
                          <Clock className="h-3 w-3 mr-1" />
                          {template.estimatedTime}
                        </div>
                      </div>
                    </div>
                    
                    <CardTitle className="text-xl font-semibold text-gray-900">{template.title}</CardTitle>
                    <CardDescription className="text-gray-600">{template.description}</CardDescription>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <div className="flex items-center flex-wrap gap-2">
                      {template.steps.map((stepType, index) => (
                        <div key={stepType} className="flex items-center">
                          <Badge variant="outline" className="text-xs px-2 py-1">
                            {agentConfig[stepType].title}
                          </Badge>
                          {index < template.steps.length - 1 && (
                            <ArrowRight className="h-3 w-3 mx-2 text-gray-400" />
                          )}
                        </div>
                      ))}
                    </div>
                    
                    <Link href="/workflows" className="block">
                      <Button className="w-full group-hover:bg-blue-600 transition-colors">
                        Start Workflow
                        <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                      </Button>
                    </Link>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </section>

        {/* AI Agents Grid */}
        <section>
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">AI Data Science Agents</h2>
            <p className="text-lg text-gray-600">Individual agent workspaces for specialized data science tasks</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {Object.entries(agentConfig).map(([agentType, config]) => {
              const Icon = config.icon
              const sessionCount = agentStats[agentType as AgentType] || 0
              const recentSession = getSessionsByAgent(agentType as AgentType)[0]
              
              return (
                <Card key={agentType} className={cn(
                  "group relative overflow-hidden transition-all duration-300 border-0",
                  config.isReady ? "hover:shadow-2xl bg-white/70 backdrop-blur-sm cursor-pointer" : "bg-gray-50 opacity-75"
                )}>
                  <div className={cn(
                    "absolute inset-0 opacity-0 transition-opacity duration-300",
                    config.isReady && "group-hover:opacity-5",
                    config.gradient
                  )} />
                  
                  <CardHeader className="relative z-10">
                    <div className="flex items-start justify-between mb-4">
                      <div className={cn('p-4 rounded-2xl shadow-sm', config.iconBg)}>
                        <Icon className="h-8 w-8" />
                      </div>
                      
                      <div className="flex flex-col items-end space-y-2">
                        {config.isReady ? (
                          <Badge variant="secondary" className="bg-green-100 text-green-700">
                            Ready
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="bg-yellow-50 text-yellow-700 border-yellow-200">
                            Coming Soon
                          </Badge>
                        )}
                        {sessionCount > 0 && (
                          <Badge variant="outline" className="text-xs">
                            {sessionCount} sessions
                          </Badge>
                        )}
                      </div>
                    </div>
                    
                    <div className="mb-2">
                      <CardTitle className="text-xl font-semibold text-gray-900 mb-1">
                        {config.title}
                      </CardTitle>
                      <div className="text-xs text-gray-500 font-mono bg-gray-100 px-2 py-1 rounded inline-block">
                        Port {config.port}
                      </div>
                    </div>
                    <CardDescription className="text-gray-600 leading-relaxed">
                      {config.description}
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-2">
                      {config.capabilities.map((capability) => (
                        <div key={capability} className="text-xs bg-gray-50 text-gray-600 px-3 py-2 rounded-lg text-center font-medium">
                          {capability}
                        </div>
                      ))}
                    </div>
                    
                    {recentSession && (
                      <div className="text-xs text-gray-500 bg-blue-50 px-3 py-2 rounded border-l-2 border-blue-200">
                        Last used: {formatRelativeTime(recentSession.createdAt)}
                      </div>
                    )}
                    
                    <Link href={config.isReady ? `/agents/${agentType}` : '#'} className="block">
                      <Button 
                        className={cn(
                          "w-full transition-all duration-300",
                          config.isReady 
                            ? "group-hover:bg-blue-600 group-hover:scale-105" 
                            : "opacity-50 cursor-not-allowed"
                        )}
                        disabled={!config.isReady}
                      >
                        {config.isReady ? "Open Workspace" : "Coming in Week 2"}
                        {config.isReady && <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />}
                      </Button>
                    </Link>
                  </CardContent>
                </Card>
              )
            })}
          </div>
        </section>

        {/* Recent Sessions */}
        {sessions.length > 0 && (
          <section>
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Recent Sessions</h2>
              <Link href="#" className="text-blue-600 hover:text-blue-700 font-medium">
                View All Sessions
                <ArrowRight className="ml-1 h-4 w-4 inline" />
              </Link>
            </div>
            
            <Card className="bg-white/70 backdrop-blur-sm border-0 shadow-lg">
              <CardContent className="p-0">
                <div className="divide-y divide-gray-100">
                  {recentSessions.map((session) => {
                    const config = agentConfig[session.agentType]
                    const Icon = config.icon
                    
                    return (
                      <Link key={session.sessionId} href={`/sessions/${session.sessionId}` as any}>
                        <div className="p-6 hover:bg-gray-50/50 transition-colors group">
                          <div className="flex items-center space-x-4">
                            <div className={cn('p-3 rounded-xl', config.iconBg)}>
                              <Icon className="h-5 w-5" />
                            </div>
                            
                            <div className="flex-1">
                              <div className="flex items-center space-x-3 mb-1">
                                <h3 className="font-semibold text-gray-900">{config.title}</h3>
                                <Badge 
                                  variant={
                                    session.status === 'completed' ? 'secondary' : 
                                    session.status === 'failed' ? 'destructive' : 'default'
                                  }
                                  className="text-xs"
                                >
                                  {session.status}
                                </Badge>
                              </div>
                              
                              <p className="text-sm text-gray-600">
                                {session.description || `Session ${session.sessionId.slice(0, 8)}...`}
                              </p>
                            </div>
                            
                            <div className="text-right text-sm text-gray-500">
                              <div className="font-medium">{formatRelativeTime(session.createdAt)}</div>
                              {session.executionTimeSeconds && (
                                <div className="text-xs">{session.executionTimeSeconds.toFixed(1)}s</div>
                              )}
                            </div>
                            
                            <ArrowRight className="h-5 w-5 text-gray-400 group-hover:text-gray-600 group-hover:translate-x-1 transition-all" />
                          </div>
                        </div>
                      </Link>
                    )
                  })}
                </div>
              </CardContent>
            </Card>
          </section>
        )}

        {/* Empty state for new users */}
        {sessions.length === 0 && (
          <section className="text-center py-16">
            <div className="max-w-md mx-auto">
              <div className="text-6xl mb-6">🚀</div>
              <h3 className="text-2xl font-bold text-gray-900 mb-4">Welcome to AI Data Science Platform</h3>
              <p className="text-gray-600 mb-8 leading-relaxed">
                Get started by choosing a workflow template above or opening an individual agent workspace to begin your data science journey.
              </p>
              <div className="space-y-3">
                <Link href="/agents/loading">
                  <Button className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3">
                    Start with Data Loading
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <p className="text-xs text-gray-500">Upload your first dataset to get started</p>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  )
}