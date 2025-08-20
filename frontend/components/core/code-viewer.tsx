'use client'

import { useState } from 'react'
import { Copy, Download, Check, Eye, EyeOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { cn, copyToClipboard } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'

interface CodeViewerProps {
  code: string
  language?: 'python' | 'sql' | 'javascript' | 'json'
  title?: string
  description?: string
  downloadFileName?: string
  className?: string
  showLineNumbers?: boolean
  maxHeight?: string
}

const languageMap = {
  python: 'py',
  sql: 'sql', 
  javascript: 'js',
  json: 'json',
}

export function CodeViewer({
  code,
  language = 'python',
  title = 'Generated Code',
  description,
  downloadFileName,
  className,
  showLineNumbers = true,
  maxHeight = '400px',
}: CodeViewerProps) {
  const [copied, setCopied] = useState(false)
  const [showCode, setShowCode] = useState(true)
  const { toast } = useToast()

  const handleCopy = async () => {
    const success = await copyToClipboard(code)
    if (success) {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
      toast({
        title: "Code copied!",
        description: "The code has been copied to your clipboard.",
        variant: "default",
      })
    } else {
      toast({
        title: "Failed to copy",
        description: "Could not copy code to clipboard.",
        variant: "destructive",
      })
    }
  }

  const handleDownload = () => {
    if (!downloadFileName) return

    const extension = languageMap[language] || 'txt'
    const filename = downloadFileName.endsWith(`.${extension}`) 
      ? downloadFileName 
      : `${downloadFileName}.${extension}`
    
    const blob = new Blob([code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
    
    toast({
      title: "File downloaded!",
      description: `Code saved as ${filename}`,
      variant: "default",
    })
  }

  const lines = code.split('\n')
  const maxLineNumber = lines.length
  const lineNumberWidth = Math.max(2, String(maxLineNumber).length)

  const getLanguageClass = (lang: string) => {
    switch (lang) {
      case 'python':
        return 'language-python'
      case 'sql':
        return 'language-sql'
      case 'javascript':
        return 'language-javascript'
      case 'json':
        return 'language-json'
      default:
        return 'language-text'
    }
  }

  // Basic syntax highlighting for Python (simple regex-based)
  const highlightPython = (code: string) => {
    return code
      .replace(/(def|class|import|from|if|else|elif|for|while|try|except|finally|with|return|yield|lambda|and|or|not|in|is|True|False|None)\b/g, '<span class="text-blue-600 font-medium">$1</span>')
      .replace(/(print|len|str|int|float|list|dict|set|tuple|range|enumerate|zip|map|filter|sorted|sum|max|min|abs|round)\b/g, '<span class="text-green-600">$1</span>')
      .replace(/(['"])((?:\\.|[^\\])*?)\1/g, '<span class="text-amber-600">$1$2$1</span>')
      .replace(/(#.*$)/gm, '<span class="text-gray-500 italic">$1</span>')
      .replace(/(\b\d+\.?\d*\b)/g, '<span class="text-purple-600">$1</span>')
  }

  const highlightCode = (code: string, lang: string) => {
    switch (lang) {
      case 'python':
        return highlightPython(code)
      default:
        return code
    }
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div className="space-y-1">
          <CardTitle className="text-lg font-medium flex items-center space-x-2">
            <span>{title}</span>
            <span className="text-xs bg-muted px-2 py-1 rounded uppercase font-normal">
              {language}
            </span>
          </CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
          <div className="text-sm text-muted-foreground">
            {lines.length} lines • {code.length} characters
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowCode(!showCode)}
          >
            {showCode ? (
              <>
                <EyeOff className="mr-2 h-4 w-4" />
                Hide
              </>
            ) : (
              <>
                <Eye className="mr-2 h-4 w-4" />
                Show
              </>
            )}
          </Button>
          
          <Button variant="outline" size="sm" onClick={handleCopy}>
            {copied ? (
              <>
                <Check className="mr-2 h-4 w-4 text-green-600" />
                Copied
              </>
            ) : (
              <>
                <Copy className="mr-2 h-4 w-4" />
                Copy
              </>
            )}
          </Button>
          
          {downloadFileName && (
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <Download className="mr-2 h-4 w-4" />
              Download
            </Button>
          )}
        </div>
      </CardHeader>
      
      {showCode && (
        <CardContent>
          <div 
            className="relative rounded-md border bg-slate-950 text-slate-50 overflow-auto"
            style={{ maxHeight }}
          >
            <div className="flex">
              {/* Line numbers */}
              {showLineNumbers && (
                <div 
                  className="flex-shrink-0 select-none bg-slate-800/50 text-slate-400 text-right py-4 px-3 text-sm font-mono border-r border-slate-800"
                  style={{ minWidth: `${lineNumberWidth + 1}ch` }}
                >
                  {lines.map((_, index) => (
                    <div key={index} className="leading-6">
                      {index + 1}
                    </div>
                  ))}
                </div>
              )}
              
              {/* Code content */}
              <div className="flex-1 overflow-auto">
                <pre className="p-4 text-sm font-mono leading-6">
                  <code 
                    className={getLanguageClass(language)}
                  >
                    {code}
                  </code>
                </pre>
              </div>
            </div>
            
            {/* Copy button overlay */}
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 p-2 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-slate-100 transition-colors opacity-0 group-hover:opacity-100"
              title="Copy code"
            >
              {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
            </button>
          </div>
          
          {/* Code statistics */}
          <div className="mt-4 flex items-center justify-between text-xs text-muted-foreground">
            <div className="space-x-4">
              <span>Format: {language.toUpperCase()}</span>
              <span>Size: {(code.length / 1024).toFixed(1)} KB</span>
            </div>
            
            {downloadFileName && (
              <span>Download as: {downloadFileName}</span>
            )}
          </div>
        </CardContent>
      )}
    </Card>
  )
}
