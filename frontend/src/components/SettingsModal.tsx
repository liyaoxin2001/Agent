import { useState, useEffect } from 'react'
import { X, Save, Trash2, AlertTriangle } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import { useTheme } from '../hooks/useTheme'

interface SettingsModalProps {
  isOpen: boolean
  onClose: () => void
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { theme } = useTheme()
  const { conversations, clearAllConversations } = useChatStore()
  const [showClearConfirm, setShowClearConfirm] = useState(false)

  if (!isOpen) return null

  const handleClearAll = () => {
    if (confirm('⚠️ 确定要清空所有对话吗？此操作不可恢复！')) {
      clearAllConversations()
      setShowClearConfirm(false)
      alert('✅ 所有对话已清空')
      onClose()
    }
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className={`rounded-lg w-full max-w-md mx-4 max-h-[90vh] overflow-hidden flex flex-col ${
        theme === 'dark' ? 'bg-gray-800' : 'bg-white'
      }`}>
        {/* 头部 */}
        <div className={`flex items-center justify-between p-4 border-b ${
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <h2 className={`text-lg font-semibold ${
            theme === 'dark' ? 'text-white' : 'text-gray-900'
          }`}>设置</h2>
          <button
            onClick={onClose}
            className={`p-1 rounded transition-colors ${
              theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
            }`}
          >
            <X size={20} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
          </button>
        </div>

        {/* 内容 */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* 数据管理 */}
          <div>
            <h3 className={`text-sm font-semibold mb-3 ${
              theme === 'dark' ? 'text-white' : 'text-gray-900'
            }`}>数据管理</h3>
            
            <div className="space-y-3">
              <div className={`rounded-lg p-3 ${
                theme === 'dark' ? 'bg-gray-700/50' : 'bg-gray-100'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-sm ${
                    theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
                  }`}>对话历史</span>
                  <span className={`text-sm ${
                    theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
                  }`}>{conversations.length} 个会话</span>
                </div>
                <p className={`text-xs mb-3 ${
                  theme === 'dark' ? 'text-gray-500' : 'text-gray-600'
                }`}>
                  所有对话数据存储在浏览器本地，不会上传到服务器
                </p>
                {!showClearConfirm ? (
                  <button
                    onClick={() => setShowClearConfirm(true)}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors text-sm"
                  >
                    <Trash2 size={16} />
                    清空所有对话
                  </button>
                ) : (
                  <div className="space-y-2">
                    <div className="flex items-start gap-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded text-xs text-yellow-400">
                      <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
                      <span>此操作将删除所有对话记录，且无法恢复。请谨慎操作！</span>
                    </div>
                    <div className="flex gap-2">
                      <button
                        onClick={handleClearAll}
                        className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors text-sm"
                      >
                        <Trash2 size={16} />
                        确认清空
                      </button>
                      <button
                        onClick={() => setShowClearConfirm(false)}
                        className={`flex-1 px-4 py-2 rounded-lg transition-colors text-sm ${
                          theme === 'dark'
                            ? 'bg-gray-600 hover:bg-gray-700 text-white'
                            : 'bg-gray-200 hover:bg-gray-300 text-gray-900'
                        }`}
                      >
                        取消
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* 关于 */}
          <div>
            <h3 className={`text-sm font-semibold mb-3 ${
              theme === 'dark' ? 'text-white' : 'text-gray-900'
            }`}>关于</h3>
            <div className={`rounded-lg p-3 space-y-2 ${
              theme === 'dark' ? 'bg-gray-700/50' : 'bg-gray-100'
            }`}>
              <div className={`text-sm ${
                theme === 'dark' ? 'text-gray-300' : 'text-gray-700'
              }`}>
                <div className="font-medium mb-1">HuahuaChat</div>
                <div className={`text-xs ${
                  theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
                }`}>
                  智能对话助手 - 支持通用对话、知识库问答、联网搜索
                </div>
              </div>
              <div className={`text-xs pt-2 border-t ${
                theme === 'dark'
                  ? 'text-gray-500 border-gray-600'
                  : 'text-gray-600 border-gray-300'
              }`}>
                <div>版本: 2.0.0</div>
                <div className="mt-1">数据存储: 浏览器本地存储 (localStorage)</div>
              </div>
            </div>
          </div>
        </div>

        {/* 底部 */}
        <div className={`p-4 border-t flex justify-end ${
          theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
        }`}>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors text-sm"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  )
}
