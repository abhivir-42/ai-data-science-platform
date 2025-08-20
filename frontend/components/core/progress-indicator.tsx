'use client'

import { useEffect, useState } from 'react'
import { AlertCircle, CheckCircle, Clock, Loader2, Play, Pause } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { cn, formatDuration } from '@/lib/utils'

export type ProgressStatus = 'idle' | 'running' | 'success' | 'error' | 'paused'

interface ProgressIndicatorProps {
  status: ProgressStatus
  progress?: number // 0-100
  message?: string
  eta?: string | number // ETA in seconds or formatted string
  startTime?: string | Date
  className?: string
  showDetails?: boolean
  variant?: 'default' | 'compact' | 'detailed'
}

const statusConfig = {
  idle: {
    icon: Clock,
    color: 'text-muted-foreground',
    bgColor: 'bg-muted',
    label: 'Ready',
    badgeVariant: 'secondary' as const,
  },
  running: {
    icon: Loader2,
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
    label: 'Running',
    badgeVariant: 'default' as const,
  },
  success: {
    icon: CheckCircle,
    color: 'text-green-600',
    bgColor: 'bg-green-50',
    label: 'Completed',
    badgeVariant: 'secondary' as const,
  },
  error: {
    icon: AlertCircle,
    color: 'text-red-600',
    bgColor: 'bg-red-50',
    label: 'Error',
    badgeVariant: 'destructive' as const,
  },
  paused: {
    icon: Pause,
    color: 'text-amber-600',
    bgColor: 'bg-amber-50',
    label: 'Paused',
    badgeVariant: 'secondary' as const,
  },
}

export function ProgressIndicator({
  status,
  progress,
  message,
  eta,
  startTime,
  className,
  showDetails = true,
  variant = 'default',
}: ProgressIndicatorProps) {
  const [elapsedTime, setElapsedTime] = useState<number>(0)
  const config = statusConfig[status]
  const Icon = config.icon

  // Calculate elapsed time
  useEffect(() => {
    if (status === 'running' && startTime) {
      const startMs = new Date(startTime).getTime()
      const interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startMs) / 1000))
      }, 1000)
      return () => clearInterval(interval)
    }
  }, [status, startTime])

  // Format ETA
  const formatETA = (eta: string | number | undefined) => {
    if (!eta) return null
    if (typeof eta === 'string') return eta
    return formatDuration(eta)
  }

  const progressValue = progress ?? (status === 'success' ? 100 : status === 'running' ? undefined : 0)

  if (variant === 'compact') {
    return (
      <div className={cn('flex items-center space-x-2', className)}>
        <Icon 
          className={cn('h-4 w-4', config.color, status === 'running' && 'animate-spin')} 
        />
        <Badge variant={config.badgeVariant}>{config.label}</Badge>
        {progress !== undefined && (
          <span className="text-sm text-muted-foreground">{Math.round(progress)}%</span>
        )}
        {message && (
          <span className="text-sm text-muted-foreground truncate max-w-48">{message}</span>
        )}
      </div>
    )
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="flex flex-row items-center space-y-0 pb-4">
        <div className="flex items-center space-x-3 flex-1">
          <div className={cn('p-2 rounded-full', config.bgColor)}>
            <Icon 
              className={cn('h-5 w-5', config.color, status === 'running' && 'animate-spin')} 
            />
          </div>
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="font-medium">{config.label}</span>
              {progressValue !== undefined && (
                <Badge variant="outline">{Math.round(progressValue)}%</Badge>
              )}
            </div>
            {message && (
              <CardDescription className="text-sm">{message}</CardDescription>
            )}
          </div>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Progress bar */}
        {progressValue !== undefined && (
          <div className="space-y-2">
            <Progress 
              value={progressValue} 
              className={cn(
                'h-2',
                status === 'error' && '[&>div]:bg-destructive',
                status === 'success' && '[&>div]:bg-secondary',
                status === 'running' && '[&>div]:bg-primary'
              )}
            />
            {showDetails && (
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{Math.round(progressValue)}% complete</span>
                {status === 'running' && progressValue < 100 && (
                  <span>{100 - Math.round(progressValue)}% remaining</span>
                )}
              </div>
            )}
          </div>
        )}

        {/* Time information */}
        {showDetails && (status === 'running' || status === 'success') && (
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center space-x-4">
              {elapsedTime > 0 && (
                <div className="flex items-center space-x-1">
                  <Clock className="h-3 w-3" />
                  <span>Elapsed: {formatDuration(elapsedTime)}</span>
                </div>
              )}
              
              {eta && status === 'running' && (
                <div className="flex items-center space-x-1">
                  <Play className="h-3 w-3" />
                  <span>ETA: {formatETA(eta)}</span>
                </div>
              )}
            </div>

            {startTime && (
              <span className="text-xs">
                Started: {new Date(startTime).toLocaleTimeString()}
              </span>
            )}
          </div>
        )}

        {/* Status-specific content */}
        {variant === 'detailed' && (
          <div className="text-xs text-muted-foreground space-y-1">
            {status === 'running' && (
              <div>
                Operation in progress... This may take a few minutes for large datasets or complex models.
              </div>
            )}
            {status === 'success' && (
              <div>
                Operation completed successfully. You can now view the results.
              </div>
            )}
            {status === 'error' && (
              <div className="text-destructive">
                Operation failed. Please check the logs for more details or try again.
              </div>
            )}
            {status === 'paused' && (
              <div>
                Operation is paused. You can resume it or start a new operation.
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
