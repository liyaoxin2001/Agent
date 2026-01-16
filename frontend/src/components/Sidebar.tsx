import { useEffect, useState, useRef } from 'react'
import {
  MessageSquare,
  Database,
  Plus,
  Trash2,
  ChevronLeft,
  Settings,
  FolderPlus,
  Upload,
  X,
  FileText,
  List,
  Cpu,
  Sun,
  Moon,
} from 'lucide-react'
import { useChatStore, ChatMode } from '../store/chatStore'
import { kbAPI, chatAPI } from '../api/client'
import DocumentManager from './DocumentManager'
import SettingsModal from './SettingsModal'
import { useTheme } from '../hooks/useTheme'

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
}

export default function Sidebar({ isOpen, onToggle }: SidebarProps) {
  const {
    currentMode,
    setMode,
    knowledgeBases,
    selectedKB,
    setSelectedKB,
    setKnowledgeBases,
    clearMessages,
    selectedModel,
    setSelectedModel,
    messages, // 获取消息列表用于显示对话
    conversations,
    currentConversationId,
    createConversation,
    switchConversation,
    deleteConversation,
    clearAllConversations,
  } = useChatStore()
  
  // 主题切换
  const { theme, toggleTheme } = useTheme()
  const [logoError, setLogoError] = useState(false)
  
  // 获取clearImageContext方法
  const { clearImageContext } = useChatStore.getState()
  
  // 清空对话时也清空图片上下文
  const handleClearMessages = () => {
    clearMessages()
    clearImageContext()
  }
  
  // 创建新会话
  const handleNewConversation = () => {
    createConversation()
  }
  
  // 切换会话
  const handleSwitchConversation = (conversationId: string) => {
    switchConversation(conversationId)
  }
  
  // 删除会话
  const handleDeleteConversation = (e: React.MouseEvent, conversationId: string) => {
    e.stopPropagation() // 阻止触发切换会话
    if (confirm('确定要删除这个会话吗？')) {
      deleteConversation(conversationId)
    }
  }

  const [showCreateKB, setShowCreateKB] = useState(false)
  const [newKBName, setNewKBName] = useState('')
  const [newKBDesc, setNewKBDesc] = useState('')
  const [creating, setCreating] = useState(false)
  const [uploading, setUploading] = useState<string | null>(null) // 当前上传的知识库名称
  const [showDocumentManager, setShowDocumentManager] = useState<string | null>(null) // 显示文档管理的知识库
  const [showSettings, setShowSettings] = useState(false) // 显示设置弹窗
  const [defaultModel, setDefaultModel] = useState<string>('gpt-3.5-turbo') // 默认模型名称
  const fileInputRef = useRef<HTMLInputElement>(null)
  const kbFileInputRefs = useRef<{ [key: string]: HTMLInputElement | null }>({})

  // 加载知识库列表和默认模型
  useEffect(() => {
    loadKnowledgeBases()
    loadDefaultModel()
  }, [])
  
  // 加载默认模型信息
  const loadDefaultModel = async () => {
    try {
      const modelInfo = await chatAPI.getDefaultModel()
      setDefaultModel(modelInfo.model_name)
    } catch (error) {
      console.error('加载默认模型失败:', error)
      // 如果失败，使用默认值
      setDefaultModel('gpt-3.5-turbo')
    }
  }

  const loadKnowledgeBases = async () => {
    try {
      const data = await kbAPI.list()
      setKnowledgeBases(data.knowledge_bases)
    } catch (error) {
      console.error('加载知识库失败:', error)
    }
  }

  // 创建知识库
  const handleCreateKB = async () => {
    if (!newKBName.trim()) return

    setCreating(true)
    try {
      await kbAPI.create(newKBName.trim(), newKBDesc.trim() || undefined)
      setNewKBName('')
      setNewKBDesc('')
      setShowCreateKB(false)
      await loadKnowledgeBases()
      // 自动选择新创建的知识库
      setSelectedKB(newKBName.trim())
      setMode('rag')
    } catch (error: any) {
      alert(`创建失败: ${error.message || '未知错误'}`)
    } finally {
      setCreating(false)
    }
  }

  // 删除知识库
  const handleDeleteKB = async (kbName: string) => {
    if (!confirm(`确定要删除知识库 "${kbName}" 吗？此操作不可恢复。`)) return

    try {
      await kbAPI.delete(kbName)
      if (selectedKB === kbName) {
        setSelectedKB(null)
      }
      await loadKnowledgeBases()
    } catch (error: any) {
      alert(`删除失败: ${error.message || '未知错误'}`)
    }
  }

  // 上传文件到知识库
  const handleKBFileUpload = async (kbName: string, file: File) => {
    setUploading(kbName)
    try {
      // 显示上传进度提示
      const fileSize = (file.size / 1024 / 1024).toFixed(2) // MB
      console.log(`📤 开始上传文件: ${file.name} (${fileSize} MB)`)
      
      const result = await kbAPI.upload(kbName, file)
      await loadKnowledgeBases() // 刷新知识库列表以更新文档数量
      
      const chunkCount = result.data?.chunk_count || 0
      alert(`✅ 文件 "${file.name}" 上传成功！\n共生成 ${chunkCount} 个文档块`)
    } catch (error: any) {
      console.error('文件上传失败:', error)
      const errorMsg = error.response?.data?.detail || error.message || '未知错误'
      alert(`❌ 上传失败: ${errorMsg}\n\n请检查：\n1. 文件格式是否支持（.txt, .pdf, .md, .docx）\n2. 文件是否损坏\n3. 网络连接是否正常`)
    } finally {
      setUploading(null)
      // 清空文件输入
      const input = kbFileInputRefs.current[kbName]
      if (input) {
        input.value = ''
      }
    }
  }

  // 处理文件选择
  const handleFileSelect = (kbName: string, e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleKBFileUpload(kbName, file)
    }
  }

  const modes: Array<{ id: ChatMode; icon: any; label: string; desc: string }> = [
    { id: 'general', icon: MessageSquare, label: '通用对话', desc: '像 ChatGPT 一样对话' },
    { id: 'rag', icon: Database, label: '知识库问答', desc: '基于文档精准回答' },
  ]

  return (
    <div
      className={`${
        isOpen ? 'w-64' : 'w-0'
      } transition-all duration-300 flex flex-col border-r overflow-hidden`}
      style={{
        backgroundColor: 'var(--bg-sidebar)',
        borderColor: 'var(--border-color)'
      }}
    >
      {/* 头部 */}
      <div className={`p-4 border-b flex items-center justify-between ${
        theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
      }`}>
        <h1 className={`text-lg font-semibold flex items-center gap-2 ${
          theme === 'dark' ? 'text-white' : 'text-gray-900'
        }`}>
          <div className="relative w-6 h-6 rounded-full overflow-hidden flex-shrink-0">
            {logoError ? (
              <span className="text-xl">🤖</span>
            ) : (
              <img 
                src="/logo.png" 
                alt="HuahuaChat Logo" 
                className="w-full h-full object-cover"
                onError={() => {
                  console.log('Sidebar logo加载失败，使用默认emoji')
                  setLogoError(true)
                }}
                onLoad={() => {
                  console.log('Sidebar logo加载成功')
                }}
              />
            )}
          </div>
          HuahuaChat
        </h1>
        <button
          onClick={onToggle}
          className={`p-1 rounded transition-colors ${
            theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
          }`}
          title="收起侧边栏"
        >
          <ChevronLeft size={20} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
        </button>
      </div>

      {/* 新对话按钮 */}
      <div className="p-4">
        <button
          onClick={handleNewConversation}
          className={`w-full flex items-center gap-2 px-4 py-3 rounded-lg transition-colors ${
            theme === 'dark'
              ? 'bg-chat-bg-light hover:bg-gray-700 text-white'
              : 'bg-gray-100 hover:bg-gray-200 text-gray-900'
          }`}
        >
          <Plus size={18} className={theme === 'dark' ? 'text-white' : 'text-gray-900'} />
          <span className={theme === 'dark' ? 'text-white' : 'text-gray-900'}>新对话</span>
        </button>
      </div>

      {/* 会话列表 */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        <h2 className={`text-xs uppercase mb-2 ${
          theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
        }`}>对话历史</h2>
        <div className="space-y-1">
          {conversations.length === 0 ? (
            <div className={`text-sm text-center py-4 ${
              theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
            }`}>暂无对话</div>
          ) : (
            conversations.map((conv) => (
              <div
                key={conv.id}
                onClick={() => handleSwitchConversation(conv.id)}
                className={`group relative flex items-center justify-between px-3 py-2.5 rounded-lg transition-all cursor-pointer ${
                  currentConversationId === conv.id
                    ? theme === 'dark'
                      ? 'bg-chat-bg-light text-white'
                      : 'bg-gray-200 text-gray-900'
                    : theme === 'dark'
                      ? 'text-gray-400 hover:bg-gray-800 hover:text-white'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">{conv.title}</div>
                  <div className={`text-xs mt-0.5 ${
                    theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                  }`}>
                    {conv.messages.filter(m => m.role === 'user' || m.role === 'assistant').length} 条消息
                  </div>
                </div>
                <button
                  onClick={(e) => handleDeleteConversation(e, conv.id)}
                  className={`opacity-0 group-hover:opacity-100 p-1 rounded-full hover:bg-red-600 transition-all ${
                    theme === 'dark' ? 'text-gray-400 hover:text-white' : 'text-gray-500 hover:text-white'
                  }`}
                  title="删除会话"
                >
                  <X size={14} />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* 模式选择 */}
      <div className="px-4 pb-4">
        <h2 className={`text-xs uppercase mb-2 ${
          theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
        }`}>对话模式</h2>
        <div className="space-y-1">
          {modes.map((mode) => (
            <button
              key={mode.id}
              onClick={() => setMode(mode.id)}
              className={`w-full flex items-start gap-3 px-3 py-2.5 rounded-lg transition-all ${
                currentMode === mode.id
                  ? theme === 'dark'
                    ? 'bg-chat-bg-light text-white'
                    : 'bg-gray-200 text-gray-900'
                  : theme === 'dark'
                    ? 'text-gray-400 hover:bg-gray-800 hover:text-white'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
              }`}
            >
              <mode.icon size={18} className={`mt-0.5 flex-shrink-0 ${
                currentMode === mode.id
                  ? theme === 'dark' ? 'text-white' : 'text-gray-900'
                  : theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
              }`} />
              <div className="text-left flex-1">
                <div className={`text-sm font-medium ${
                  currentMode === mode.id
                    ? theme === 'dark' ? 'text-white' : 'text-gray-900'
                    : theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
                }`}>{mode.label}</div>
                <div className={`text-xs ${
                  theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                }`}>{mode.desc}</div>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* 模型选择（仅通用模式） */}
      {currentMode === 'general' && (
        <div className="px-4 pb-4">
          <h2 className={`text-xs uppercase mb-2 flex items-center gap-2 ${
            theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
          }`}>
            <Cpu size={12} />
            模型选择
          </h2>
          <select
            value={selectedModel || ''}
            onChange={(e) => setSelectedModel(e.target.value || null)}
            className={`w-full px-3 py-2 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 border ${
              theme === 'dark'
                ? 'bg-gray-800 text-white border-gray-700'
                : 'bg-white text-gray-900 border-gray-300'
            }`}
          >
            <option value="">默认模型: {defaultModel}</option>
            <optgroup label="支持视觉的模型">
              <option value="gpt-4o">GPT-4o（最新，推荐）</option>
              <option value="gpt-4-turbo">GPT-4 Turbo</option>
              <option value="gpt-4o-mini">GPT-4o Mini（轻量）</option>
              <option value="gpt-4-vision-preview">GPT-4 Vision Preview</option>
            </optgroup>
            <optgroup label="基础模型（不支持视觉）">
              <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
              <option value="gpt-3.5-turbo-16k">GPT-3.5 Turbo 16K</option>
            </optgroup>
          </select>
          <div className={`text-xs mt-1 ${
            theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
          }`}>
            {selectedModel ? (
              <span>当前: {selectedModel}</span>
            ) : (
              <span>使用默认模型: {defaultModel}</span>
            )}
          </div>
        </div>
      )}

      {/* 知识库选择（仅 RAG 模式） */}
      {currentMode === 'rag' && (
        <div className="px-4 pb-4 flex-1 overflow-y-auto">
          <div className="flex items-center justify-between mb-2">
            <h2 className={`text-xs uppercase ${
              theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
            }`}>知识库</h2>
            <button
              onClick={() => setShowCreateKB(!showCreateKB)}
              className={`p-1 rounded transition-colors ${
                theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
              }`}
              title="创建知识库"
            >
              <FolderPlus size={14} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
            </button>
          </div>

          {/* 创建知识库表单 */}
          {showCreateKB && (
            <div className={`mb-3 p-3 rounded-lg space-y-2 ${
              theme === 'dark' ? 'bg-gray-800' : 'bg-gray-100'
            }`}>
              <input
                type="text"
                placeholder="知识库名称"
                value={newKBName}
                onChange={(e) => setNewKBName(e.target.value)}
                className={`w-full px-2 py-1.5 rounded text-sm focus:outline-none focus:ring-1 focus:ring-purple-500 ${
                  theme === 'dark'
                    ? 'bg-gray-700 text-white'
                    : 'bg-white text-gray-900 border border-gray-300'
                }`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleCreateKB()
                  if (e.key === 'Escape') setShowCreateKB(false)
                }}
              />
              <input
                type="text"
                placeholder="描述（可选）"
                value={newKBDesc}
                onChange={(e) => setNewKBDesc(e.target.value)}
                className={`w-full px-2 py-1.5 rounded text-sm focus:outline-none focus:ring-1 focus:ring-purple-500 ${
                  theme === 'dark'
                    ? 'bg-gray-700 text-white'
                    : 'bg-white text-gray-900 border border-gray-300'
                }`}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleCreateKB()
                }}
              />
              <div className="flex gap-2">
                <button
                  onClick={handleCreateKB}
                  disabled={creating || !newKBName.trim()}
                  className="flex-1 px-2 py-1.5 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 rounded text-sm transition-colors text-white"
                >
                  {creating ? '创建中...' : '创建'}
                </button>
                <button
                  onClick={() => {
                    setShowCreateKB(false)
                    setNewKBName('')
                    setNewKBDesc('')
                  }}
                  className={`px-2 py-1.5 rounded text-sm transition-colors ${
                    theme === 'dark'
                      ? 'bg-gray-700 hover:bg-gray-600 text-white'
                      : 'bg-gray-200 hover:bg-gray-300 text-gray-900'
                  }`}
                >
                  取消
                </button>
              </div>
            </div>
          )}

          {/* 知识库列表 */}
          <div className="space-y-1">
            {knowledgeBases.map((kb) => (
              <div
                key={kb.name}
                className={`group flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
                  selectedKB === kb.name
                    ? theme === 'dark'
                      ? 'bg-chat-bg-light text-white'
                      : 'bg-gray-200 text-gray-900'
                    : theme === 'dark'
                      ? 'text-gray-400 hover:bg-gray-800 hover:text-white'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                }`}
              >
                <button
                  onClick={() => setSelectedKB(kb.name)}
                  className="flex-1 text-left"
                >
                  <div className="text-sm font-medium">{kb.name}</div>
                  <div className={`text-xs ${
                    theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
                  }`}>{kb.document_count} 个文档</div>
                </button>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {/* 查看文档按钮 */}
                  <button
                    onClick={() => setShowDocumentManager(kb.name)}
                    className="p-1 hover:bg-green-600 rounded transition-all"
                    title="查看文档列表"
                  >
                    <List size={14} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
                  </button>
                  {/* 上传文件按钮 */}
                  <input
                    ref={(el) => (kbFileInputRefs.current[kb.name] = el)}
                    type="file"
                    accept=".txt,.pdf,.md,.docx"
                    className="hidden"
                    onChange={(e) => handleFileSelect(kb.name, e)}
                    disabled={uploading === kb.name}
                  />
                  <button
                    onClick={() => kbFileInputRefs.current[kb.name]?.click()}
                    disabled={uploading === kb.name}
                    className="p-1 hover:bg-blue-600 rounded transition-all disabled:opacity-50"
                    title="上传文件到此知识库"
                  >
                    {uploading === kb.name ? (
                      <div className="w-3 h-3 border border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <Upload size={14} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
                    )}
                  </button>
                  {/* 删除按钮 */}
                  <button
                    onClick={() => handleDeleteKB(kb.name)}
                    className="p-1 hover:bg-red-600 rounded transition-all"
                    title="删除知识库"
                  >
                    <X size={14} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
                  </button>
                </div>
              </div>
            ))}
            {knowledgeBases.length === 0 && !showCreateKB && (
              <div className={`text-sm text-center py-4 ${
                theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
              }`}>
                暂无知识库
                <br />
                <button
                  onClick={() => setShowCreateKB(true)}
                  className="text-purple-400 hover:text-purple-300 mt-1"
                >
                  点击创建
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* 底部设置 */}
      <div className={`p-4 border-t space-y-2 ${
        theme === 'dark' ? 'border-gray-700' : 'border-gray-200'
      }`}>
        {/* 主题切换按钮 */}
        <button
          onClick={toggleTheme}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
            theme === 'dark'
              ? 'text-gray-400 hover:bg-gray-800 hover:text-white'
              : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900'
          }`}
          title={theme === 'dark' ? '切换到亮色模式' : '切换到暗色模式'}
        >
          {theme === 'dark' ? (
            <>
              <Sun size={16} />
              <span className="text-sm">亮色模式</span>
            </>
          ) : (
            <>
              <Moon size={16} />
              <span className="text-sm">暗色模式</span>
            </>
          )}
        </button>
        
        <button
          onClick={() => {
            if (confirm('⚠️ 确定要清空所有对话吗？此操作不可恢复！')) {
              clearAllConversations()
              alert('✅ 所有对话已清空')
            }
          }}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
            theme === 'dark'
              ? 'text-gray-400 hover:bg-gray-800 hover:text-white'
              : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900'
          }`}
        >
          <Trash2 size={16} />
          <span className="text-sm">清空所有对话</span>
        </button>
        <button
          onClick={() => setShowSettings(true)}
          className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg transition-all ${
            theme === 'dark'
              ? 'text-gray-400 hover:bg-gray-800 hover:text-white'
              : 'text-gray-600 hover:bg-gray-200 hover:text-gray-900'
          }`}
        >
          <Settings size={16} />
          <span className="text-sm">设置</span>
        </button>
      </div>

      {/* 设置弹窗 */}
      <SettingsModal
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />

      {/* 文档管理弹窗 */}
      {showDocumentManager && (
        <DocumentManager
          kbName={showDocumentManager}
          onClose={() => setShowDocumentManager(null)}
          onUpdate={loadKnowledgeBases}
        />
      )}
    </div>
  )
}
