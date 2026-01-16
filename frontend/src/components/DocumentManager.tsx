import { useState, useEffect } from 'react'
import { X, FileText, Trash2, Eye } from 'lucide-react'
import { kbAPI } from '../api/client'
import { useTheme } from '../hooks/useTheme'

interface Document {
  filename: string
  source: string
  chunk_count: number
  upload_time?: string
  metadata?: any
}

interface DocumentManagerProps {
  kbName: string
  onClose: () => void
  onUpdate: () => void
}

export default function DocumentManager({ kbName, onClose, onUpdate }: DocumentManagerProps) {
  const { theme } = useTheme()
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [deleting, setDeleting] = useState<string | null>(null)

  useEffect(() => {
    loadDocuments()
  }, [kbName])

  const loadDocuments = async () => {
    setLoading(true)
    try {
      const data = await kbAPI.listDocuments(kbName)
      setDocuments(data.documents || [])
    } catch (error: any) {
      console.error('加载文档列表失败:', error)
      alert(`加载文档列表失败: ${error.message || '未知错误'}`)
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (filename: string) => {
    if (!confirm(`确定要删除文档 "${filename}" 吗？此操作不可恢复。`)) return

    setDeleting(filename)
    try {
      await kbAPI.deleteDocument(kbName, filename)
      await loadDocuments()
      onUpdate() // 通知父组件更新
    } catch (error: any) {
      alert(`删除失败: ${error.message || '未知错误'}`)
    } finally {
      setDeleting(null)
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className={`rounded-lg w-full max-w-2xl max-h-[80vh] flex flex-col ${
        theme === 'dark' ? 'bg-sidebar-bg' : 'bg-white'
      }`}>
        {/* 头部 */}
        <div className={`p-4 border-b flex items-center justify-between ${
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <h2 className={`text-lg font-semibold ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>文档管理 - {kbName}</h2>
          <button
            onClick={onClose}
            className={`p-1 rounded transition-colors ${
              theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
            }`}
          >
            <X size={20} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
          </button>
        </div>

        {/* 文档列表 */}
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className={`text-center py-8 ${
              theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
            }`}>加载中...</div>
          ) : documents.length === 0 ? (
            <div className={`text-center py-8 ${
              theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
            }`}>暂无文档</div>
          ) : (
            <div className="space-y-2">
              {documents.map((doc) => (
                <div
                  key={doc.source}
                  className={`flex items-center justify-between p-3 rounded-lg transition-colors ${
                    theme === 'dark'
                      ? 'bg-gray-800 hover:bg-gray-700'
                      : 'bg-gray-100 hover:bg-gray-200'
                  }`}
                >
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <FileText size={18} className={`flex-shrink-0 ${
                      theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
                    }`} />
                    <div className="flex-1 min-w-0">
                      <div className={`text-sm font-medium truncate ${
                        theme === 'dark' ? 'text-white' : 'text-gray-900'
                      }`}>{doc.filename}</div>
                      <div className={`text-xs ${
                        theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                      }`}>
                        {doc.chunk_count} 个分块
                        {doc.upload_time && ` • ${new Date(doc.upload_time).toLocaleString()}`}
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <button
                      onClick={() => handleDelete(doc.filename)}
                      disabled={deleting === doc.filename}
                      className="p-1.5 hover:bg-red-600 rounded transition-colors disabled:opacity-50"
                      title="删除文档"
                    >
                      {deleting === doc.filename ? (
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <Trash2 size={16} />
                      )}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 底部 */}
        <div className={`p-4 border-t flex justify-between items-center ${
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <div className={`text-sm ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            共 {documents.length} 个文档
          </div>
          <button
            onClick={onClose}
            className={`px-4 py-2 rounded-lg transition-colors ${
              theme === 'dark'
                ? 'bg-gray-700 hover:bg-gray-600 text-white'
                : 'bg-gray-200 hover:bg-gray-300 text-gray-900'
            }`}
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  )
}
