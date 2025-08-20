'use client'

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, X, AlertCircle, CheckCircle, FileText, FileSpreadsheet } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { cn, formatFileSize, isValidFileType, fileToBase64 } from '@/lib/utils'
import { useToast } from '@/hooks/use-toast'

interface UploadedFile {
  file: File
  id: string
  status: 'pending' | 'uploading' | 'completed' | 'error'
  progress?: number
  base64?: string
  error?: string
}

interface FileUploaderProps {
  accept?: string[]
  multiple?: boolean
  maxSize?: number // in bytes
  maxFiles?: number
  onUpload: (files: UploadedFile[]) => void
  onFileProcess?: (file: UploadedFile) => Promise<void>
  className?: string
  title?: string
  description?: string
  disabled?: boolean
}

const defaultAcceptedTypes = ['csv', 'xlsx', 'xls', 'json', 'pdf', 'parquet', 'tsv']

const getFileIcon = (filename: string) => {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'csv':
    case 'tsv':
      return FileText
    case 'xlsx':
    case 'xls':
      return FileSpreadsheet
    case 'json':
      return FileText
    case 'pdf':
      return File
    case 'parquet':
      return File
    default:
      return File
  }
}

const getFileTypeLabel = (filename: string) => {
  const ext = filename.split('.').pop()?.toLowerCase()
  switch (ext) {
    case 'csv':
      return 'CSV'
    case 'xlsx':
    case 'xls':
      return 'Excel'
    case 'json':
      return 'JSON'
    case 'pdf':
      return 'PDF'
    case 'parquet':
      return 'Parquet'
    case 'tsv':
      return 'TSV'
    default:
      return ext?.toUpperCase() || 'Unknown'
  }
}

export function FileUploader({
  accept = defaultAcceptedTypes,
  multiple = false,
  maxSize = 16 * 1024 * 1024, // 16MB default
  maxFiles = multiple ? 10 : 1,
  onUpload,
  onFileProcess,
  className,
  title = 'Upload Dataset',
  description = 'Drag and drop your files here, or click to select files',
  disabled = false,
}: FileUploaderProps) {
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([])
  const { toast } = useToast()

  const processFile = useCallback(async (file: File): Promise<UploadedFile> => {
    const id = Math.random().toString(36).substring(2) + Date.now().toString(36)
    const uploadedFile: UploadedFile = {
      file,
      id,
      status: 'pending',
    }

    try {
      // Validate file
      if (!isValidFileType(file.name, accept)) {
        throw new Error(`File type not supported. Accepted types: ${accept.join(', ')}`)
      }

      if (file.size > maxSize) {
        throw new Error(`File too large. Maximum size: ${formatFileSize(maxSize)}`)
      }

      // Update status to uploading
      setUploadedFiles(prev => 
        prev.map(f => f.id === id ? { ...f, status: 'uploading' as const, progress: 0 } : f)
      )

      // Convert to base64
      uploadedFile.status = 'uploading'
      uploadedFile.progress = 50

      setUploadedFiles(prev => 
        prev.map(f => f.id === id ? { ...f, progress: 50 } : f)
      )

      const base64 = await fileToBase64(file)
      uploadedFile.base64 = base64
      uploadedFile.progress = 100
      uploadedFile.status = 'completed'

      // Custom processing if provided
      if (onFileProcess) {
        await onFileProcess(uploadedFile)
      }

      setUploadedFiles(prev => 
        prev.map(f => f.id === id ? { ...f, ...uploadedFile } : f)
      )

      toast({
        title: "File processed successfully",
        description: `${file.name} is ready for analysis`,
        variant: "default",
      })

      return uploadedFile
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      uploadedFile.status = 'error'
      uploadedFile.error = errorMessage

      setUploadedFiles(prev => 
        prev.map(f => f.id === id ? { ...f, ...uploadedFile } : f)
      )

      toast({
        title: "File processing failed",
        description: errorMessage,
        variant: "destructive",
      })

      return uploadedFile
    }
  }, [accept, maxSize, onFileProcess, toast])

  const onDrop = useCallback(async (acceptedFiles: File[], rejectedFiles: any[]) => {
    // Handle rejected files
    rejectedFiles.forEach(({ file, errors }) => {
      errors.forEach((error: any) => {
        toast({
          title: "File rejected",
          description: `${file.name}: ${error.message}`,
          variant: "destructive",
        })
      })
    })

    // Check max files limit
    if (uploadedFiles.length + acceptedFiles.length > maxFiles) {
      toast({
        title: "Too many files",
        description: `Maximum ${maxFiles} file${maxFiles > 1 ? 's' : ''} allowed`,
        variant: "destructive",
      })
      return
    }

    // Add files to state first
    const newFiles = acceptedFiles.map(file => ({
      file,
      id: Math.random().toString(36).substring(2) + Date.now().toString(36),
      status: 'pending' as const,
    }))

    setUploadedFiles(prev => [...prev, ...newFiles])

    // Process files
    const processedFiles = await Promise.all(
      newFiles.map(({ file }) => processFile(file))
    )

    // Call onUpload with completed files
    const completedFiles = processedFiles.filter(f => f.status === 'completed')
    if (completedFiles.length > 0) {
      onUpload(completedFiles)
    }
  }, [uploadedFiles.length, maxFiles, processFile, onUpload, toast])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/json': ['.json'],
      'application/pdf': ['.pdf'],
      'application/octet-stream': ['.parquet'],
      'text/tab-separated-values': ['.tsv'],
    },
    multiple,
    maxSize,
    disabled,
  })

  const removeFile = (id: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== id))
  }

  const clearAll = () => {
    setUploadedFiles([])
  }

  const completedFiles = uploadedFiles.filter(f => f.status === 'completed')
  const hasErrors = uploadedFiles.some(f => f.status === 'error')
  const isUploading = uploadedFiles.some(f => f.status === 'uploading')

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>{title}</span>
          {uploadedFiles.length > 0 && (
            <Button variant="outline" size="sm" onClick={clearAll}>
              Clear All
            </Button>
          )}
        </CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Drop zone */}
        <div
          {...getRootProps()}
          className={cn(
            'border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors',
            isDragActive && !isDragReject && 'border-primary bg-primary/5',
            isDragReject && 'border-destructive bg-destructive/5',
            disabled && 'cursor-not-allowed opacity-50',
            !isDragActive && !isDragReject && 'border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/25'
          )}
        >
          <input {...getInputProps()} />
          
          <div className="flex flex-col items-center space-y-4">
            <div className={cn(
              'p-4 rounded-full',
              isDragActive && !isDragReject && 'bg-primary/10',
              isDragReject && 'bg-destructive/10',
              !isDragActive && 'bg-muted/50'
            )}>
              <Upload className={cn(
                'h-8 w-8',
                isDragActive && !isDragReject && 'text-primary',
                isDragReject && 'text-destructive',
                !isDragActive && 'text-muted-foreground'
              )} />
            </div>
            
            <div className="space-y-2">
              {isDragReject ? (
                <p className="text-destructive font-medium">Invalid file type</p>
              ) : isDragActive ? (
                <p className="text-primary font-medium">Drop files here</p>
              ) : (
                <div className="space-y-1">
                  <p className="font-medium">Drag and drop files here, or click to browse</p>
                  <p className="text-sm text-muted-foreground">
                    Supported formats: {accept.join(', ').toUpperCase()}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Max file size: {formatFileSize(maxSize)}
                    {multiple && ` • Max files: ${maxFiles}`}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* File list */}
        {uploadedFiles.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <h4 className="text-sm font-medium">
                Files ({uploadedFiles.length})
              </h4>
              {completedFiles.length > 0 && (
                <span className="text-xs text-muted-foreground">
                  {completedFiles.length} ready for analysis
                </span>
              )}
            </div>
            
            <div className="space-y-2">
              {uploadedFiles.map((uploadedFile) => {
                const Icon = getFileIcon(uploadedFile.file.name)
                return (
                  <div
                    key={uploadedFile.id}
                    className="flex items-center space-x-3 p-3 rounded-lg border bg-muted/25"
                  >
                    <div className={cn(
                      'p-2 rounded',
                      uploadedFile.status === 'completed' && 'bg-green-100 text-green-600',
                      uploadedFile.status === 'error' && 'bg-red-100 text-red-600',
                      uploadedFile.status === 'uploading' && 'bg-blue-100 text-blue-600',
                      uploadedFile.status === 'pending' && 'bg-gray-100 text-gray-600'
                    )}>
                      <Icon className="h-4 w-4" />
                    </div>
                    
                    <div className="flex-1 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium truncate">
                          {uploadedFile.file.name}
                        </span>
                        <div className="flex items-center space-x-2">
                          <span className="text-xs text-muted-foreground">
                            {getFileTypeLabel(uploadedFile.file.name)}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {formatFileSize(uploadedFile.file.size)}
                          </span>
                          {uploadedFile.status === 'completed' && (
                            <CheckCircle className="h-4 w-4 text-green-600" />
                          )}
                          {uploadedFile.status === 'error' && (
                            <AlertCircle className="h-4 w-4 text-red-600" />
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => removeFile(uploadedFile.id)}
                            className="h-6 w-6 p-0"
                          >
                            <X className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                      
                      {uploadedFile.status === 'uploading' && (
                        <Progress 
                          value={uploadedFile.progress || 0} 
                          className="h-1"
                        />
                      )}
                      
                      {uploadedFile.error && (
                        <p className="text-xs text-destructive">
                          {uploadedFile.error}
                        </p>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Status summary */}
        {uploadedFiles.length > 0 && (
          <div className="flex items-center justify-between text-sm text-muted-foreground border-t pt-4">
            <div className="space-x-4">
              {isUploading && (
                <span className="text-blue-600">Processing files...</span>
              )}
              {hasErrors && (
                <span className="text-destructive">
                  {uploadedFiles.filter(f => f.status === 'error').length} error(s)
                </span>
              )}
              {completedFiles.length > 0 && (
                <span className="text-green-600">
                  {completedFiles.length} ready
                </span>
              )}
            </div>
            
            <span>
              Total size: {formatFileSize(
                uploadedFiles.reduce((total, f) => total + f.file.size, 0)
              )}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
