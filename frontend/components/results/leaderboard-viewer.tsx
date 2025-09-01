'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Trophy, Medal, Award, Download, Eye, BarChart3, Zap, Clock, Target, Gauge } from 'lucide-react'
import { cn } from '@/lib/utils'

interface LeaderboardEntry {
  model_id: string
  auc?: number
  logloss?: number
  rmse?: number
  mean_residual_deviance?: number
  accuracy?: number
  f1?: number
  precision?: number
  recall?: number
  mae?: number
  [key: string]: any
}

interface LeaderboardViewerProps {
  leaderboard: LeaderboardEntry[]
  bestModelId?: string
  onModelSelect?: (modelId: string) => void
  onDownloadModel?: (modelId: string) => void
  className?: string
}

export function LeaderboardViewer({ 
  leaderboard, 
  bestModelId, 
  onModelSelect, 
  onDownloadModel, 
  className 
}: LeaderboardViewerProps) {
  const [sortBy, setSortBy] = useState<string>('default')
  const [sortedLeaderboard, setSortedLeaderboard] = useState<LeaderboardEntry[]>(leaderboard)

  useEffect(() => {
    setSortedLeaderboard(leaderboard)
  }, [leaderboard])

  const getRankIcon = (index: number) => {
    switch (index) {
      case 0:
        return <Trophy className="h-5 w-5 text-yellow-500" />
      case 1:
        return <Medal className="h-5 w-5 text-gray-400" />
      case 2:
        return <Award className="h-5 w-5 text-amber-600" />
      default:
        return <span className="text-lg font-bold text-gray-400">#{index + 1}</span>
    }
  }

  const getPerformanceColor = (value: number, metric: string) => {
    // For metrics where lower is better (logloss, rmse, mae)
    const lowerIsBetter = ['logloss', 'rmse', 'mae', 'mean_residual_deviance']
    
    if (lowerIsBetter.includes(metric)) {
      if (value <= 0.1) return 'text-green-600'
      if (value <= 0.3) return 'text-yellow-600'
      return 'text-red-600'
    } else {
      // For metrics where higher is better (auc, accuracy, f1, precision, recall)
      if (value >= 0.9) return 'text-green-600'
      if (value >= 0.7) return 'text-yellow-600'
      return 'text-red-600'
    }
  }

  const formatMetricValue = (value: number) => {
    return value?.toFixed(4) || 'N/A'
  }

  const getMetricIcon = (metric: string) => {
    switch (metric) {
      case 'auc':
        return <BarChart3 className="h-4 w-4" />
      case 'accuracy':
        return <Target className="h-4 w-4" />
      case 'logloss':
        return <Zap className="h-4 w-4" />
      case 'rmse':
      case 'mae':
        return <Gauge className="h-4 w-4" />
      default:
        return <Clock className="h-4 w-4" />
    }
  }

  if (!leaderboard || leaderboard.length === 0) {
    return (
      <Card className={className}>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Trophy className="h-5 w-5 text-yellow-500" />
            Model Leaderboard
          </CardTitle>
          <CardDescription>No models available in leaderboard</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  // Get available metrics
  const availableMetrics = Object.keys(leaderboard[0] || {}).filter(key => 
    key !== 'model_id' && typeof leaderboard[0][key] === 'number'
  )

  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Trophy className="h-5 w-5 text-yellow-500" />
          Model Performance Leaderboard
        </CardTitle>
        <CardDescription>
          Trained {leaderboard.length} models • Best model: {bestModelId || 'Not specified'}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {/* Top 3 Models Highlight */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {leaderboard.slice(0, 3).map((model, index) => (
              <Card key={model.model_id} className={cn(
                "relative border-2",
                index === 0 && "border-yellow-200 bg-yellow-50",
                index === 1 && "border-gray-200 bg-gray-50",
                index === 2 && "border-amber-200 bg-amber-50"
              )}>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getRankIcon(index)}
                      <Badge variant={index === 0 ? "default" : "secondary"}>
                        {index === 0 ? "Champion" : index === 1 ? "Runner-up" : "3rd Place"}
                      </Badge>
                    </div>
                    {bestModelId === model.model_id && (
                      <Badge variant="outline" className="text-green-600 border-green-600">
                        Best
                      </Badge>
                    )}
                  </div>
                  <div className="text-sm font-mono text-gray-600">
                    {model.model_id}
                  </div>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="space-y-2">
                    {availableMetrics.slice(0, 3).map(metric => (
                      <div key={metric} className="flex items-center justify-between">
                        <div className="flex items-center gap-1 text-sm">
                          {getMetricIcon(metric)}
                          <span className="capitalize">{metric}</span>
                        </div>
                        <span className={cn(
                          "font-bold text-sm",
                          getPerformanceColor(model[metric], metric)
                        )}>
                          {formatMetricValue(model[metric])}
                        </span>
                      </div>
                    ))}
                  </div>
                  
                  <div className="flex gap-2 mt-4">
                    {onModelSelect && (
                      <Button 
                        size="sm" 
                        variant="outline" 
                        onClick={() => onModelSelect(model.model_id)}
                        className="flex-1"
                      >
                        <Eye className="h-3 w-3 mr-1" />
                        View
                      </Button>
                    )}
                    {onDownloadModel && (
                      <Button 
                        size="sm" 
                        variant="outline" 
                        onClick={() => onDownloadModel(model.model_id)}
                        className="flex-1"
                      >
                        <Download className="h-3 w-3 mr-1" />
                        Save
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Complete Leaderboard Table */}
          {leaderboard.length > 3 && (
            <div>
              <h3 className="text-lg font-semibold mb-3">Complete Rankings</h3>
              <div className="border rounded-lg overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">Rank</TableHead>
                      <TableHead>Model ID</TableHead>
                      {availableMetrics.map(metric => (
                        <TableHead key={metric} className="text-center">
                          <div className="flex items-center justify-center gap-1">
                            {getMetricIcon(metric)}
                            <span className="capitalize">{metric}</span>
                          </div>
                        </TableHead>
                      ))}
                      <TableHead className="w-24">Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {leaderboard.map((model, index) => (
                      <TableRow key={model.model_id} className={cn(
                        index < 3 && "bg-gray-50",
                        bestModelId === model.model_id && "bg-green-50 border-green-200"
                      )}>
                        <TableCell className="font-medium">
                          <div className="flex items-center gap-2">
                            {index < 3 ? getRankIcon(index) : <span>#{index + 1}</span>}
                          </div>
                        </TableCell>
                        <TableCell className="font-mono text-sm">
                          <div className="flex items-center gap-2">
                            {model.model_id}
                            {bestModelId === model.model_id && (
                              <Badge variant="outline" className="text-green-600 border-green-600 text-xs">
                                Best
                              </Badge>
                            )}
                          </div>
                        </TableCell>
                        {availableMetrics.map(metric => (
                          <TableCell key={metric} className="text-center">
                            <span className={cn(
                              "font-medium",
                              getPerformanceColor(model[metric], metric)
                            )}>
                              {formatMetricValue(model[metric])}
                            </span>
                          </TableCell>
                        ))}
                        <TableCell>
                          <div className="flex gap-1">
                            {onModelSelect && (
                              <Button 
                                size="sm" 
                                variant="ghost" 
                                onClick={() => onModelSelect(model.model_id)}
                              >
                                <Eye className="h-3 w-3" />
                              </Button>
                            )}
                            {onDownloadModel && (
                              <Button 
                                size="sm" 
                                variant="ghost" 
                                onClick={() => onDownloadModel(model.model_id)}
                              >
                                <Download className="h-3 w-3" />
                              </Button>
                            )}
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}

          {/* Performance Summary */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Performance Summary</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-900">{leaderboard.length}</div>
                  <div className="text-sm text-gray-600">Models Trained</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {availableMetrics.length}
                  </div>
                  <div className="text-sm text-gray-600">Metrics Tracked</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {bestModelId ? '1' : '0'}
                  </div>
                  <div className="text-sm text-gray-600">Best Model</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {Math.round((leaderboard.filter(m => m.auc && m.auc > 0.8).length / leaderboard.length) * 100)}%
                  </div>
                  <div className="text-sm text-gray-600">High Performance</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </CardContent>
    </Card>
  )
}
