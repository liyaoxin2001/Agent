import { useState, useEffect } from 'react'
import { MessageSquare, Database } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import { useTheme } from '../hooks/useTheme'

export default function EmptyState() {
  const { currentMode, enableSearch, setInputValue } = useChatStore()
  const { theme } = useTheme()
  const [logoError, setLogoError] = useState(false)
  const [logoUrl, setLogoUrl] = useState('/logo.png')
  
  // 每次组件加载时更新logo URL，避免缓存
  useEffect(() => {
    const timestamp = Date.now()
    const newUrl = `/logo.png?t=${timestamp}`
    console.log('尝试加载logo:', newUrl)
    setLogoUrl(newUrl)
    setLogoError(false) // 重置错误状态
    
    // 预加载图片检查
    const img = new Image()
    img.onload = () => {
      console.log('Logo预加载成功，图片存在')
      setLogoError(false)
    }
    img.onerror = () => {
      console.log('Logo预加载失败，文件可能不存在，URL:', newUrl)
      setLogoError(true)
    }
    img.src = newUrl
  }, [])

  const examples = {
    general: [
      '写一首关于春天的诗',
      '解释一下量子计算的基本原理',
      '帮我写一个 Python 快速排序算法',
    ],
    rag: [
      '这份文档的主要内容是什么？',
      'Python 有哪些特点？',
      '如何使用这个API？',
    ],
  }

  const icons = {
    general: MessageSquare,
    rag: Database,
  }

  const Icon = icons[currentMode]

  return (
    <div className="flex flex-col items-center justify-center h-full px-4 text-center">
      <div className="mb-8">
        <div className={`relative inline-flex items-center justify-center w-16 h-16 rounded-full mb-4 overflow-hidden ${
          logoError ? 'bg-purple-600' : ''
        }`}>
          {/* 默认图标（作为后备） */}
          {logoError && (
            <Icon size={32} className="text-white" />
          )}
          {/* Logo图片 */}
          {!logoError && (
            <img 
              src={logoUrl}
              alt="HuahuaChat Logo" 
              className="w-full h-full object-cover"
              style={{ position: 'absolute', top: 0, left: 0 }}
              onError={(e) => {
                console.error('Logo图片加载失败:', e)
                console.error('失败的URL:', logoUrl)
                setLogoError(true)
              }}
              onLoad={() => {
                console.log('✅ Logo图片显示成功，URL:', logoUrl)
              }}
            />
          )}
        </div>
        <h2 className={`text-2xl font-bold mb-2 ${
          theme === 'dark' ? 'text-white' : 'text-gray-900'
        }`}>HuahuaChat</h2>
        <p className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'}>
          {currentMode === 'general' && (
            <>
              通用对话模式 - 像 ChatGPT 一样对话
              {enableSearch && (
                <span className="block mt-1 text-purple-400">
                  🔍 联网搜索已启用，可获取最新信息
                </span>
              )}
            </>
          )}
          {currentMode === 'rag' && '知识库问答 - 基于文档精准回答'}
        </p>
      </div>

      <div className="max-w-2xl w-full">
        <h3 className={`text-sm mb-4 ${
          theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
        }`}>示例问题：</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {examples[currentMode].map((example, i) => (
            <button
              key={i}
              className="p-4 rounded-lg text-left transition-all transform hover:scale-105"
              style={{
                backgroundColor: 'var(--bg-chat-light)',
                color: 'var(--text-primary)'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.opacity = '0.8'
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.opacity = '1'
              }}
              onClick={() => {
                setInputValue(example)
              }}
            >
              <p className="text-sm">{example}</p>
            </button>
          ))}
        </div>
      </div>

      <div className={`mt-12 text-xs ${
        theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
      }`}>
        <p>💡 提示：Enter 发送，Shift+Enter 换行</p>
        <p className="mt-1">
          📎 点击左下角按钮上传文件（支持 txt, pdf, md, docx）
        </p>
        {currentMode === 'general' && (
          <p className="mt-1">
            🔍 点击搜索按钮启用联网搜索，获取最新信息
          </p>
        )}
      </div>
    </div>
  )
}
