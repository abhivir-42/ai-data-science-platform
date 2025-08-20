'use client'

import { useEffect, useRef, useState } from 'react'
import dynamic from 'next/dynamic'
import { Download, Maximize2, Minimize2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

interface PlotlyChartProps {
  figure: any // Plotly figure object
  title?: string
  description?: string
  className?: string
  height?: number
  width?: number
  exportEnabled?: boolean
  fullscreenEnabled?: boolean
  config?: Partial<Plotly.Config>
}

const defaultConfig: Partial<Plotly.Config> = {
  displayModeBar: true,
  displaylogo: false,
  modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
  responsive: true,
  toImageButtonOptions: {
    format: 'png',
    filename: 'chart',
    height: 800,
    width: 1200,
    scale: 2,
  },
}

export function PlotlyChart({
  figure,
  title = 'Visualization',
  description,
  className,
  height = 400,
  width,
  exportEnabled = true,
  fullscreenEnabled = true,
  config = {},
}: PlotlyChartProps) {
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [plotLoaded, setPlotLoaded] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  const mergedConfig = {
    ...defaultConfig,
    ...config,
  }

  const handleExport = async (format: 'png' | 'svg' | 'pdf' | 'html') => {
    try {
      if (format === 'html') {
        // Export as HTML
        const plotlyScript = `
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <title>${title}</title>
</head>
<body>
    <div id="plotly-div" style="height: 100vh; width: 100%;"></div>
    <script>
        Plotly.newPlot('plotly-div', ${JSON.stringify(figure.data)}, ${JSON.stringify(figure.layout)});
    </script>
</body>
</html>`
        
        const blob = new Blob([plotlyScript], { type: 'text/html' })
        const url = URL.createObjectURL(blob)
        const link = document.createElement('a')
        link.href = url
        link.download = `${title.toLowerCase().replace(/\s+/g, '-')}.html`
        document.body.appendChild(link)
        link.click()
        document.body.removeChild(link)
        URL.revokeObjectURL(url)
        
        toast({
          title: "Chart exported!",
          description: `Chart saved as HTML file.`,
          variant: "default",
        })
      } else {
        // Use Plotly's built-in export (handled by the toolbar)
        toast({
          title: "Export initiated",
          description: `Use the camera icon in the chart toolbar to export as ${format.toUpperCase()}.`,
          variant: "default",
        })
      }
    } catch (error) {
      console.error('Export error:', error)
      toast({
        title: "Export failed",
        description: "Could not export the chart.",
        variant: "destructive",
      })
    }
  }

  const toggleFullscreen = () => {
    if (isFullscreen) {
      setIsFullscreen(false)
      if (document.exitFullscreen) {
        document.exitFullscreen()
      }
    } else {
      setIsFullscreen(true)
      if (containerRef.current && containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen()
      }
    }
  }

  // Handle fullscreen change events
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  // Validate figure data
  if (!figure || !figure.data || !Array.isArray(figure.data)) {
    return (
      <Card className={cn('w-full', className)}>
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-64 text-muted-foreground">
            <div className="text-center">
              <div className="text-lg mb-2">📊</div>
              <div>No chart data available</div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  const chartHeight = isFullscreen ? '100vh' : height
  const chartWidth = isFullscreen ? '100vw' : width

  return (
    <Card className={cn('w-full', className)} ref={containerRef}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div className="space-y-1">
          <CardTitle className="text-lg font-medium">{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
          <div className="text-sm text-muted-foreground">
            {figure.data.length} trace{figure.data.length !== 1 ? 's' : ''}
            {figure.layout?.title && ` • ${figure.layout.title.text || figure.layout.title}`}
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {exportEnabled && (
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => handleExport('html')}
            >
              <Download className="mr-2 h-4 w-4" />
              Export
            </Button>
          )}
          
          {fullscreenEnabled && (
            <Button 
              variant="outline" 
              size="sm"
              onClick={toggleFullscreen}
            >
              {isFullscreen ? (
                <>
                  <Minimize2 className="mr-2 h-4 w-4" />
                  Exit
                </>
              ) : (
                <>
                  <Maximize2 className="mr-2 h-4 w-4" />
                  Full
                </>
              )}
            </Button>
          )}
        </div>
      </CardHeader>
      
      <CardContent className={cn('p-0', isFullscreen && 'h-screen')}>
        <div className={cn('plotly-chart', isFullscreen && 'h-full w-full')}>
          {typeof window !== 'undefined' && (
            <Plot
              data={figure.data}
              layout={{
                ...figure.layout,
                autosize: true,
                margin: {
                  l: 60,
                  r: 60,
                  t: 60,
                  b: 60,
                  ...figure.layout?.margin,
                },
                font: {
                  family: 'Inter, system-ui, sans-serif',
                  size: 12,
                  ...figure.layout?.font,
                },
              }}
              config={mergedConfig}
              style={{
                width: chartWidth || '100%',
                height: chartHeight,
              }}
              useResizeHandler={true}
              onInitialized={() => setPlotLoaded(true)}
              onError={(error) => {
                console.error('Plotly error:', error)
                toast({
                  title: "Chart error",
                  description: "Failed to render the chart.",
                  variant: "destructive",
                })
              }}
            />
          )}
          
          {!plotLoaded && typeof window !== 'undefined' && (
            <div className="flex items-center justify-center" style={{ height: chartHeight }}>
              <div className="text-center text-muted-foreground">
                <div className="animate-spin-slow text-2xl mb-2">📊</div>
                <div>Loading chart...</div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
      
      {plotLoaded && (
        <div className="px-6 pb-4">
          <div className="text-xs text-muted-foreground">
            Interactive chart • Hover for details • Use toolbar to zoom, pan, and export
          </div>
        </div>
      )}
    </Card>
  )
}
