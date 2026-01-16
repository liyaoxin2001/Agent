import { useState } from 'react'
import { User, Bot } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Message as MessageType } from '../store/chatStore'
import { useTheme } from '../hooks/useTheme'

interface MessageProps {
  message: MessageType
}

export default function Message({ message }: MessageProps) {
  const { theme } = useTheme()
  const isUser = message.role === 'user'
  const isSystem = message.role === 'system'
  const [avatarError, setAvatarError] = useState(false)

  if (isSystem) {
    return (
      <div className="flex justify-center">
        <div className="bg-blue-500/10 text-blue-400 px-4 py-2 rounded-lg text-sm">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div
      className="flex gap-4 p-6 rounded-lg animate-fade-in"
      style={{
        backgroundColor: isUser ? 'var(--bg-chat-light)' : 'var(--bg-chat-light)',
        color: 'var(--text-primary)'
      }}
    >
      {/* 头像 */}
      <div
        className={`relative flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center overflow-hidden ${
          isUser ? 'bg-purple-600' : 'bg-green-600'
        }`}
      >
        {isUser ? (
          <User size={20} className="text-white" />
        ) : (
          <>
            {avatarError ? (
              <Bot size={20} className="text-white" />
            ) : (
              <img 
                src="/logo.png" 
                alt="花花" 
                className="w-full h-full object-cover"
                onError={() => {
                  console.log('AI头像logo加载失败，使用默认Bot图标')
                  setAvatarError(true)
                }}
                onLoad={() => {
                  console.log('AI头像logo加载成功')
                }}
              />
            )}
          </>
        )}
      </div>

      {/* 消息内容 */}
      <div className="flex-1 overflow-hidden">
        <div 
          className="text-sm font-semibold mb-1"
          style={{ color: 'var(--text-primary)' }}
        >{isUser ? '你' : '花花'}</div>
        
        {/* 图片显示 */}
        {message.images && message.images.length > 0 && (
          <div className="flex gap-2 mb-2 flex-wrap">
            {message.images.map((img, index) => (
              <img
                key={index}
                src={img}
                alt={`图片 ${index + 1}`}
                className={`max-w-xs max-h-64 object-contain rounded-lg border ${
                  theme === 'dark' ? 'border-gray-600' : 'border-gray-300'
                }`}
              />
            ))}
          </div>
        )}
        
        <div className={`prose max-w-none ${
          theme === 'dark' ? 'prose-invert' : ''
        }`}>
          {isUser ? (
            <p 
              className="whitespace-pre-wrap"
              style={{ color: 'var(--text-primary)' }}
            >{message.content || (message.images ? '（图片）' : '')}</p>
          ) : (
            <ReactMarkdown
              components={{
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '')
                  return !inline && match ? (
                    <SyntaxHighlighter
                      {...props}
                      style={theme === 'dark' ? vscDarkPlus : oneLight}
                      language={match[1]}
                      PreTag="div"
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  ) : (
                    <code {...props} className={className}>
                      {children}
                    </code>
                  )
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>
      </div>
    </div>
  )
}
