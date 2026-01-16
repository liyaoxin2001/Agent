import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { Send, Paperclip, Image as ImageIcon, Search } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import { useTheme } from '../hooks/useTheme'
import { chatAPI, kbAPI } from '../api/client'

export default function ChatInput() {
  const [input, setInput] = useState('')
  const [uploading, setUploading] = useState(false)
  const [pendingImages, setPendingImages] = useState<string[]>([]) // 待发送的图片列表
  const fileInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  
  const {
    addMessage,
    currentMode,
    selectedKB,
    isLoading,
    setLoading,
    addUploadedFile,
    setSelectedKB,
    setMode,
    enableSearch,
    setEnableSearch,
    selectedModel,
    currentImageContext,
    setCurrentImageContext,
    messages, // 获取消息列表用于构建对话历史
    inputValueToSet,
    setInputValue,
  } = useChatStore()
  
  const { theme } = useTheme()
  
  // 监听 inputValueToSet，当有值时填充到输入框
  useEffect(() => {
    if (inputValueToSet !== null) {
      setInput(inputValueToSet)
      setInputValue(null) // 清空，避免重复设置
      // 自动聚焦到输入框
      setTimeout(() => {
        textareaRef.current?.focus()
      }, 100)
    }
  }, [inputValueToSet, setInputValue])

  // 处理发送消息（支持图片+文字一起发送）
  const handleSend = async () => {
    // 如果没有输入且没有图片，不发送
    if ((!input.trim() && pendingImages.length === 0) || isLoading) return

    const userMessage = input.trim() || (pendingImages.length > 0 ? '请分析这张图片' : '')
    const imagesToSend = [...pendingImages] // 复制图片列表
    
    // 只在有新图片时才发送图片，后续对话不再自动携带图片
    // 这样更符合ChatGPT的使用习惯：图片只在第一次发送时传递
    const finalImages = imagesToSend.length > 0 ? imagesToSend : []
    
    // 清空输入和待发送图片
    setInput('')
    setPendingImages([])
    
    // 如果有新图片，更新图片上下文（用于显示，但不用于后续API调用）
    if (imagesToSend.length > 0) {
      setCurrentImageContext(imagesToSend)
    } else {
      // 如果没有新图片，清空图片上下文（后续对话不再携带图片）
      setCurrentImageContext([])
    }
    
    // 在添加用户消息之前构建对话历史（不包含当前消息）
    // 注意：历史消息不携带图片，只在当前消息有新图片时才传递图片
    // 这样可以减少token消耗，也更符合ChatGPT的使用习惯
    const recentHistory = messages
      .filter(msg => msg.role === 'user' || msg.role === 'assistant')
      .slice(-10) // 只保留最近10轮对话
      .map(msg => ({
        role: msg.role,
        content: msg.content,
        images: [] // 历史消息不携带图片，只在当前消息有新图片时才传递
      }))
    
    // 添加用户消息（包含图片）
    addMessage({ 
      role: 'user', 
      content: userMessage,
      images: finalImages.length > 0 ? finalImages : undefined
    })
    setLoading(true)

    try {
      let response: string

      if (currentMode === 'rag') {
        // RAG 模式（不支持图片）
        if (finalImages.length > 0) {
          addMessage({
            role: 'system',
            content: '知识库模式不支持图片，请切换到"通用对话"模式',
          })
          setLoading(false)
          return
        }
        
        if (!selectedKB) {
          addMessage({
            role: 'system',
            content: '请先选择一个知识库或上传文件',
          })
          setLoading(false)
          return
        }
        
        const result = await chatAPI.chat({
          question: userMessage,
          kb_name: selectedKB,
          stream: false,
        })
        response = result.answer
      } else {
        // 通用模式（支持图片）
        if (enableSearch && finalImages.length === 0) {
          // 启用搜索：先搜索，再结合搜索结果回答（搜索模式不支持图片）
          try {
            const searchResult = await chatAPI.searchAndChat(userMessage)
            response = searchResult.answer
          } catch (searchError: any) {
            // 搜索失败，回退到普通对话
            console.warn('搜索失败，使用普通对话:', searchError)
            // 使用之前构建的对话历史
            response = await chatAPI.chatGeneral(
              userMessage, 
              finalImages.length > 0 ? finalImages : undefined, 
              selectedModel,
              recentHistory
            )
          }
        } else {
          // 普通对话（支持图片和模型选择）
          // 传递对话历史，让AI能理解上下文
          response = await chatAPI.chatGeneral(
            userMessage, 
            finalImages.length > 0 ? finalImages : undefined, 
            selectedModel,
            recentHistory // 传递对话历史
          )
        }
      }

      // 添加 AI 回复
      addMessage({ role: 'assistant', content: response })
    } catch (error: any) {
      console.error('发送消息失败:', error)
      addMessage({
        role: 'assistant',
        content: `抱歉，发生了错误：${error.message || error.response?.data?.detail || '未知错误'}`,
      })
    } finally {
      setLoading(false)
      // 发送消息后自动聚焦到输入框
      setTimeout(() => {
        textareaRef.current?.focus()
      }, 100)
    }
  }

  // 处理文件上传（快速上传到临时知识库）
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    
    try {
      // 使用固定的临时知识库名称
      const tempKBName = 'temp_upload'
      
      // 先尝试上传，如果知识库不存在则创建
      try {
        await kbAPI.upload(tempKBName, file)
      } catch (uploadError: any) {
        // 如果上传失败（可能是知识库不存在），先创建知识库
        if (uploadError.response?.status === 404 || uploadError.response?.status === 400) {
          await kbAPI.create(tempKBName, '临时上传文件')
          await kbAPI.upload(tempKBName, file)
        } else {
          throw uploadError
        }
      }
      
      // 切换到 RAG 模式
      setMode('rag')
      setSelectedKB(tempKBName)
      addUploadedFile(file.name)
      
      addMessage({
        role: 'system',
        content: `📎 已上传文件: ${file.name}，已切换到知识库模式`,
      })
    } catch (error: any) {
      addMessage({
        role: 'system',
        content: `上传失败: ${error.message || error.response?.data?.detail || '未知错误'}`,
      })
    } finally {
      setUploading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }
    }
  }

  // 处理图片上传（转换为base64，保存在待发送列表中，不立即发送）
  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // 检查文件类型
    if (!file.type.startsWith('image/')) {
      addMessage({
        role: 'system',
        content: '请选择图片文件',
      })
      if (imageInputRef.current) {
        imageInputRef.current.value = ''
      }
      return
    }

    setUploading(true)
    
    try {
      // 将图片转换为base64
      const reader = new FileReader()
      reader.onload = (event) => {
        try {
          const base64Data = event.target?.result as string
          
          // 添加到待发送图片列表（不立即发送，只显示在预览区域）
          setPendingImages(prev => [...prev, base64Data])
          addUploadedFile(file.name)
        } catch (error: any) {
          addMessage({
            role: 'system',
            content: `图片处理失败: ${error.message || '未知错误'}`,
          })
        } finally {
          setUploading(false)
          if (imageInputRef.current) {
            imageInputRef.current.value = ''
          }
        }
      }
      
      reader.onerror = () => {
        addMessage({
          role: 'system',
          content: '图片读取失败',
        })
        setUploading(false)
        if (imageInputRef.current) {
          imageInputRef.current.value = ''
        }
      }
      
      // 读取文件为base64
      reader.readAsDataURL(file)
    } catch (error: any) {
      addMessage({
        role: 'system',
        content: `图片上传失败: ${error.message || '未知错误'}`,
      })
      setUploading(false)
      if (imageInputRef.current) {
        imageInputRef.current.value = ''
      }
    }
  }
  
  // 移除待发送的图片
  const removePendingImage = (index: number) => {
    setPendingImages(prev => prev.filter((_, i) => i !== index))
  }

  // 处理键盘事件
  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // 自动调整文本框高度
  const adjustHeight = () => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
    }
  }

  return (
    <div 
      className="border-t transition-colors"
      style={{
        backgroundColor: 'var(--bg-chat)',
        borderColor: 'var(--border-color)'
      }}
    >
      <div className="max-w-3xl mx-auto px-4 py-4">
        {/* 工具栏 */}
        <div className="flex items-center gap-2 mb-2">
          {/* 搜索开关（仅通用模式） */}
          {currentMode === 'general' && (
            <button
              onClick={() => setEnableSearch(!enableSearch)}
              className={`p-2 rounded-lg transition-all ${
                enableSearch
                  ? 'bg-purple-600 text-white'
                  : theme === 'dark'
                    ? 'hover:bg-gray-700 text-gray-400'
                    : 'hover:bg-gray-200 text-gray-600'
              }`}
              title={enableSearch ? '关闭联网搜索' : '启用联网搜索'}
            >
              <Search size={18} className={enableSearch ? '' : 'opacity-50'} />
            </button>
          )}

          {/* 文件上传 */}
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.pdf,.md,.docx"
            className="hidden"
            onChange={handleFileUpload}
          />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={uploading}
                className={`p-2 rounded-lg transition-colors disabled:opacity-50 ${
                  theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
                }`}
                title="快速上传文件（txt, pdf, md, docx）"
              >
                <Paperclip size={18} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
              </button>

              {/* 图片上传 */}
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />
              <button
                onClick={() => imageInputRef.current?.click()}
                disabled={uploading}
                className={`p-2 rounded-lg transition-colors disabled:opacity-50 ${
                  theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
                }`}
                title="上传图片（支持 jpg, png, gif 等）"
              >
                <ImageIcon size={18} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
              </button>

          {/* 搜索状态提示 */}
          {currentMode === 'general' && enableSearch && (
            <span className="text-xs text-purple-400 ml-auto">
              🔍 联网搜索已启用
            </span>
          )}
        </div>

        {/* 输入区域 */}
        <div className="flex gap-3 items-end">
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => {
                  setInput(e.target.value)
                  adjustHeight()
                }}
                onKeyDown={handleKeyDown}
                placeholder={
                  currentMode === 'rag' && !selectedKB
                    ? '请先选择知识库或上传文件...'
                    : enableSearch
                    ? '输入消息... (已启用联网搜索，Enter 发送，Shift+Enter 换行)'
                    : '输入消息... (Enter 发送，Shift+Enter 换行)'
                }
                className="flex-1 rounded-lg px-4 py-3 resize-none focus:outline-none focus:ring-2 focus:ring-purple-500 min-h-[52px] max-h-[200px] border"
                style={{
                  backgroundColor: 'var(--bg-chat-light)',
                  color: 'var(--text-primary)',
                  borderColor: 'var(--border-color)'
                }}
                rows={1}
                disabled={isLoading || (currentMode === 'rag' && !selectedKB)}
              />
          <button
            onClick={handleSend}
            disabled={(!input.trim() && pendingImages.length === 0) || isLoading || (currentMode === 'rag' && !selectedKB)}
            className="p-3 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:opacity-50 rounded-lg transition-colors flex-shrink-0"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </button>
        </div>
        
        {/* 待发送的图片预览 */}
        {pendingImages.length > 0 && (
          <div className="flex gap-2 mt-2 flex-wrap">
            {pendingImages.map((img, index) => (
              <div key={index} className="relative group">
                <img
                  src={img}
                  alt={`预览 ${index + 1}`}
                  className={`w-20 h-20 object-cover rounded-lg border ${
                    theme === 'dark' ? 'border-gray-600' : 'border-gray-300'
                  }`}
                />
                <button
                  onClick={() => removePendingImage(index)}
                  className="absolute -top-2 -right-2 w-5 h-5 bg-red-600 hover:bg-red-700 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <span className="text-white text-xs">×</span>
                </button>
              </div>
            ))}
          </div>
        )}

        {/* 提示信息 */}
        <div className={`mt-2 text-xs text-center ${
          theme === 'dark' ? 'text-gray-500' : 'text-gray-400'
        }`}>
          {currentMode === 'rag' && '知识库模式 - 基于文档回答'}
          {currentMode === 'general' && enableSearch && '通用模式 - 联网搜索增强'}
          {currentMode === 'general' && !enableSearch && '通用模式 - 像 ChatGPT 一样对话'}
          {pendingImages.length > 0 && ` • ${pendingImages.length} 张图片待发送`}
        </div>
      </div>
    </div>
  )
}
