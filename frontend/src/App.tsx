import { useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import { useTheme } from './hooks/useTheme'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  useTheme() // 初始化主题

  return (
    <div 
      className="flex h-screen overflow-hidden transition-colors duration-200"
      style={{
        backgroundColor: 'var(--bg-chat)',
        color: 'var(--text-primary)'
      }}
    >
      {/* 侧边栏 */}
      <Sidebar isOpen={sidebarOpen} onToggle={() => setSidebarOpen(!sidebarOpen)} />
      
      {/* 主聊天区域 */}
      <div 
        className="flex-1 flex flex-col relative"
        style={{
          backgroundColor: 'var(--bg-chat)'
        }}
      >
        <ChatArea sidebarOpen={sidebarOpen} />
      </div>
    </div>
  )
}

export default App
