'use client'

import { useState, useMemo } from 'react'
import { Download, Search, ChevronLeft, ChevronRight, ArrowUpDown } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { cn, formatFileSize } from '@/lib/utils'

interface DataFrameData {
  records: Array<Record<string, unknown>>
  columns: string[]
}

interface DataFrameViewerProps {
  data: DataFrameData
  maxRows?: number
  downloadUrls?: {
    csv?: string
    json?: string
    excel?: string
  }
  onDownload?: (format: 'csv' | 'json' | 'excel') => void
  title?: string
  description?: string
  className?: string
  virtualizeRows?: boolean
}

const ROWS_PER_PAGE = 50

export function DataFrameViewer({
  data,
  maxRows = 1000,
  downloadUrls,
  onDownload,
  title = 'Dataset',
  description,
  className,
  virtualizeRows = false,
}: DataFrameViewerProps) {
  const [currentPage, setCurrentPage] = useState(0)
  const [rowsPerPage, setRowsPerPage] = useState(ROWS_PER_PAGE)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  // Filter and sort data
  const processedData = useMemo(() => {
    let filtered = data.records

    // Search filter
    if (searchTerm) {
      filtered = filtered.filter(row =>
        Object.values(row).some(value =>
          String(value).toLowerCase().includes(searchTerm.toLowerCase())
        )
      )
    }

    // Sort
    if (sortColumn) {
      filtered = [...filtered].sort((a, b) => {
        const aVal = a[sortColumn]
        const bVal = b[sortColumn]
        
        if (aVal === null || aVal === undefined) return sortDirection === 'asc' ? 1 : -1
        if (bVal === null || bVal === undefined) return sortDirection === 'asc' ? -1 : 1
        
        if (typeof aVal === 'number' && typeof bVal === 'number') {
          return sortDirection === 'asc' ? aVal - bVal : bVal - aVal
        }
        
        const aStr = String(aVal).toLowerCase()
        const bStr = String(bVal).toLowerCase()
        
        if (aStr < bStr) return sortDirection === 'asc' ? -1 : 1
        if (aStr > bStr) return sortDirection === 'asc' ? 1 : -1
        return 0
      })
    }

    return filtered
  }, [data.records, searchTerm, sortColumn, sortDirection])

  // Pagination
  const totalPages = Math.ceil(processedData.length / rowsPerPage)
  const startIndex = currentPage * rowsPerPage
  const endIndex = Math.min(startIndex + rowsPerPage, processedData.length)
  const currentPageData = processedData.slice(startIndex, endIndex)

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortColumn(column)
      setSortDirection('asc')
    }
  }

  const handleDownload = (format: 'csv' | 'json' | 'excel') => {
    // Use onDownload callback if available (new approach)
    if (onDownload) {
      onDownload(format)
      return
    }
    
    // Fallback to URL-based download (legacy approach)
    if (downloadUrls?.[format]) {
      const link = document.createElement('a')
      link.href = downloadUrls[format]!
      link.download = `${title.toLowerCase().replace(/\s+/g, '-')}.${format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  const renderCellValue = (value: unknown) => {
    if (value === null || value === undefined) {
      return <span className="text-muted-foreground italic">null</span>
    }
    
    if (typeof value === 'boolean') {
      return <span className={value ? 'text-green-600' : 'text-red-600'}>{String(value)}</span>
    }
    
    if (typeof value === 'number') {
      return <span className="font-mono">{value}</span>
    }
    
    if (typeof value === 'string' && value.length > 100) {
      return (
        <span title={value} className="cursor-help">
          {value.substring(0, 100)}...
        </span>
      )
    }
    
    return <span>{String(value)}</span>
  }

  return (
    <Card className={cn('w-full', className)}>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
        <div className="space-y-1">
          <CardTitle className="text-lg font-medium">{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
          <div className="text-sm text-muted-foreground">
            {processedData.length.toLocaleString()} rows × {data.columns.length} columns
            {processedData.length !== data.records.length && (
              <span className="ml-2">
                (filtered from {data.records.length.toLocaleString()})
              </span>
            )}
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          {(downloadUrls || onDownload) && (
            <Select onValueChange={(value) => handleDownload(value as any)}>
              <SelectTrigger className="w-32">
                <Download className="mr-2 h-4 w-4" />
                <SelectValue placeholder="Download" />
              </SelectTrigger>
              <SelectContent>
                {(downloadUrls?.csv || onDownload) && <SelectItem value="csv">CSV</SelectItem>}
                {(downloadUrls?.json || onDownload) && <SelectItem value="json">JSON</SelectItem>}
                {(downloadUrls?.excel || onDownload) && <SelectItem value="excel">Excel</SelectItem>}
              </SelectContent>
            </Select>
          )}
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Search and pagination controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
              <Input
                placeholder="Search data..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-64 pl-8"
              />
            </div>
            
            <Select 
              value={String(rowsPerPage)} 
              onValueChange={(value) => {
                setRowsPerPage(Number(value))
                setCurrentPage(0)
              }}
            >
              <SelectTrigger className="w-24">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="25">25</SelectItem>
                <SelectItem value="50">50</SelectItem>
                <SelectItem value="100">100</SelectItem>
                <SelectItem value="200">200</SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* Pagination */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground">
              {startIndex + 1}–{endIndex} of {processedData.length}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.max(0, currentPage - 1))}
              disabled={currentPage === 0}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setCurrentPage(Math.min(totalPages - 1, currentPage + 1))}
              disabled={currentPage >= totalPages - 1}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Table */}
        <div className="rounded-md border overflow-auto max-h-96">
          <table className="w-full text-sm data-table">
            <thead className="bg-muted/50">
              <tr>
                {data.columns.map((column) => (
                  <th
                    key={column}
                    className="h-10 px-4 text-left align-middle font-medium cursor-pointer hover:bg-muted/80 transition-colors"
                    onClick={() => handleSort(column)}
                  >
                    <div className="flex items-center space-x-2">
                      <span className="truncate" title={column}>
                        {column}
                      </span>
                      <ArrowUpDown className="h-4 w-4 opacity-50" />
                      {sortColumn === column && (
                        <span className="text-xs">
                          {sortDirection === 'asc' ? '↑' : '↓'}
                        </span>
                      )}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {currentPageData.map((row, index) => (
                <tr key={startIndex + index} className="border-t hover:bg-muted/25 transition-colors">
                  {data.columns.map((column) => (
                    <td key={column} className="p-4 align-middle max-w-xs">
                      {renderCellValue(row[column])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          
          {currentPageData.length === 0 && (
            <div className="py-12 text-center text-muted-foreground">
              {searchTerm ? 'No matching records found' : 'No data available'}
            </div>
          )}
        </div>

        {/* Summary stats */}
        <div className="text-xs text-muted-foreground space-y-1">
          <div>Data types: {data.columns.length} columns</div>
          {maxRows < data.records.length && (
            <div className="text-amber-600">
              Showing first {maxRows.toLocaleString()} rows of {data.records.length.toLocaleString()}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
