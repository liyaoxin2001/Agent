import { useRef, useEffect } from 'react'
import { Menu } from 'lucide-react'
import { useChatStore } from '../store/chatStore'
import { useTheme } from '../hooks/useTheme'
import Message from './Message'
import ChatInput from './ChatInput'
import EmptyState from './EmptyState'

interface ChatAreaProps {
  sidebarOpen: boolean
}

export default function ChatArea({ sidebarOpen }: ChatAreaProps) {
  const { messages } = useChatStore()
  const { theme } = useTheme()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div 
      className="flex flex-col h-full"
      style={{
        backgroundColor: 'var(--bg-chat)'
      }}
    >
      {/* 顶部导航栏（可选） */}
      <div 
        className="flex items-center px-4 py-3 border-b"
        style={{
          backgroundColor: 'var(--bg-chat)',
          borderColor: 'var(--border-color)'
        }}
      >
        {!sidebarOpen && (
          <button
            className={`p-1.5 rounded transition-colors mr-2 ${
              theme === 'dark' ? 'hover:bg-gray-700' : 'hover:bg-gray-200'
            }`}
            onClick={() => {/* 切换侧边栏 */}}
          >
            <Menu size={20} className={theme === 'dark' ? 'text-gray-400' : 'text-gray-600'} />
          </button>
        )}
        <div className={`flex-1 text-center text-sm ${
          theme === 'dark' ? 'text-gray-400' : 'text-gray-600'
        }`}>
          HuahuaChat - 智能对话助手
        </div>
      </div>

      {/* 消息区域 */}
      <div 
        className="flex-1 overflow-y-auto"
        style={{
          backgroundColor: 'var(--bg-chat)'
        }}
      >
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-4">
            {messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* 输入区域 */}
      <ChatInput />
    </div>
  )
}
