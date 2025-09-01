'use client'

import { useState, useRef } from 'react'
import { Upload, File, X, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { cn } from '@/lib/utils'

interface WorkflowFileUploaderProps {
  onFileSelect: (file: File) => void
  selectedFile: File | null
  onRemoveFile: () => void
  className?: string
  accept?: string
  maxSizeMB?: number
}

export function WorkflowFileUploader({
  onFileSelect,
  selectedFile,
  onRemoveFile,
  className,
  accept = '.csv,.xlsx,.xls,.json,.parquet',
  maxSizeMB = 100
}: WorkflowFileUploaderProps) {
  const [dragActive, setDragActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const validateFile = (file: File): string | null => {
    // Check file size
    const maxSizeBytes = maxSizeMB * 1024 * 1024
    if (file.size > maxSizeBytes) {
      return `File size must be less than ${maxSizeMB}MB`
    }

    // Check file type
    const allowedExtensions = accept.split(',').map(ext => ext.trim().toLowerCase())
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()
    
    if (!allowedExtensions.includes(fileExtension)) {
      return `File type not supported. Allowed types: ${allowedExtensions.join(', ')}`
    }

    return null
  }

  const handleFile = (file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setError(null)
    onFileSelect(file)
  }

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const openFileDialog = () => {
    fileInputRef.current?.click()
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  if (selectedFile) {
    return (
      <Card className={cn('border-2 border-green-200 bg-green-50', className)}>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-green-100 rounded-lg">
                <File className="h-5 w-5 text-green-600" />
              </div>
              <div>
                <div className="font-medium text-green-900">{selectedFile.name}</div>
                <div className="text-sm text-green-600">
                  {formatFileSize(selectedFile.size)} • Ready for workflow
                </div>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onRemoveFile}
              className="text-green-600 hover:text-green-700 hover:bg-green-100"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className={className}>
      <Card
        className={cn(
          'border-2 border-dashed border-gray-300 hover:border-gray-400 transition-colors cursor-pointer',
          dragActive && 'border-blue-500 bg-blue-50',
          error && 'border-red-300 bg-red-50'
        )}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        onClick={openFileDialog}
      >
        <CardContent className="p-8">
          <div className="text-center">
            <div className={cn(
              'mx-auto w-12 h-12 rounded-lg flex items-center justify-center mb-4',
              error ? 'bg-red-100' : 'bg-gray-100'
            )}>
              {error ? (
                <AlertCircle className="h-6 w-6 text-red-600" />
              ) : (
                <Upload className="h-6 w-6 text-gray-600" />
              )}
            </div>
            
            {error ? (
              <div className="text-red-700">
                <div className="font-medium mb-1">Upload Error</div>
                <div className="text-sm">{error}</div>
              </div>
            ) : (
              <>
                <div className="text-lg font-medium text-gray-900 mb-2">
                  Drop your file here or click to upload
                </div>
                <div className="text-sm text-gray-600 mb-4">
                  Supports CSV, Excel, JSON, and Parquet files up to {maxSizeMB}MB
                </div>
                <Button type="button" variant="outline" className="pointer-events-none">
                  <Upload className="h-4 w-4 mr-2" />
                  Choose File
                </Button>
              </>
            )}
          </div>
        </CardContent>
      </Card>

      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleInputChange}
        className="hidden"
      />
    </div>
  )
}
