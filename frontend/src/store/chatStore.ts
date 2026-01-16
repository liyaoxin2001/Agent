import { create } from 'zustand'

export type ChatMode = 'rag' | 'general'

export interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: number
  images?: string[] // base64图片数据列表
}

export interface KnowledgeBase {
  name: string
  description?: string
  document_count: number
}

export interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: number
  updatedAt: number
  mode: ChatMode
  selectedKB?: string | null
  currentImageContext: string[]
}

interface ChatState {
  // 会话管理
  conversations: Conversation[]
  currentConversationId: string | null
  
  // 当前会话的数据（从 conversations 中获取）
  messages: Message[]
  currentMode: ChatMode
  isLoading: boolean
  // 当前对话的图片上下文（用于后续对话）
  currentImageContext: string[] // 最近一次对话中的图片
  
  // 知识库相关
  knowledgeBases: KnowledgeBase[]
  selectedKB: string | null
  
  // 临时文件
  uploadedFiles: string[]
  
  // 搜索开关
  enableSearch: boolean
  
  // 模型选择
  selectedModel: string | null
  
  // 输入框值设置（用于示例问题点击）
  inputValueToSet: string | null
  
  // Actions
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => void
  clearMessages: () => void
  setMode: (mode: ChatMode) => void
  setLoading: (loading: boolean) => void
  setKnowledgeBases: (kbs: KnowledgeBase[]) => void
  setSelectedKB: (kb: string | null) => void
  addUploadedFile: (filename: string) => void
  clearUploadedFiles: () => void
  setEnableSearch: (enable: boolean) => void
  setSelectedModel: (model: string | null) => void
  setCurrentImageContext: (images: string[]) => void
  clearImageContext: () => void
  setInputValue: (value: string | null) => void // 设置输入框值
  
  // 会话管理 Actions
  createConversation: () => string // 返回新会话ID
  switchConversation: (conversationId: string) => void
  deleteConversation: (conversationId: string) => void
  updateConversationTitle: (conversationId: string, title: string) => void
  clearAllConversations: () => void // 清空所有对话
}

// 从 localStorage 加载会话
const loadConversationsFromStorage = (): Conversation[] => {
  try {
    const stored = localStorage.getItem('huahuaChat_conversations')
    if (stored) {
      return JSON.parse(stored)
    }
  } catch (error) {
    console.error('加载会话失败:', error)
  }
  return []
}

// 保存会话到 localStorage
const saveConversationsToStorage = (conversations: Conversation[]) => {
  try {
    localStorage.setItem('huahuaChat_conversations', JSON.stringify(conversations))
  } catch (error) {
    console.error('保存会话失败:', error)
  }
}

// 生成会话标题（从第一条用户消息）
const generateConversationTitle = (firstMessage: string): string => {
  // 截取前30个字符作为标题
  const title = firstMessage.trim().slice(0, 30)
  return title || '新对话'
}

export const useChatStore = create<ChatState>((set, get) => {
  // 初始化时加载会话
  const conversations = loadConversationsFromStorage()
  const currentConversationId = conversations.length > 0 ? conversations[0].id : null
  
  // 获取当前会话的数据（初始化时）
  let currentConv = {
    messages: [] as Message[],
    currentMode: 'general' as ChatMode,
    currentImageContext: [] as string[],
    selectedKB: null as string | null,
  }
  
  if (currentConversationId) {
    const conv = conversations.find(c => c.id === currentConversationId)
    if (conv) {
      currentConv = {
        messages: conv.messages,
        currentMode: conv.mode,
        currentImageContext: conv.currentImageContext,
        selectedKB: conv.selectedKB || null,
      }
    }
  }
  
  return {
    // Initial state
    conversations,
    currentConversationId,
    messages: currentConv.messages,
    currentMode: currentConv.currentMode,
    isLoading: false,
    knowledgeBases: [],
    selectedKB: currentConv.selectedKB,
    uploadedFiles: [],
    enableSearch: false,
    selectedModel: null, // null表示使用默认模型
    currentImageContext: currentConv.currentImageContext,
    inputValueToSet: null, // 用于示例问题点击
  
  // Actions
  addMessage: (message) =>
    set((state) => {
      const newMessage = {
        ...message,
        id: `msg-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
      }
      
      let updatedConversations = [...state.conversations]
      let currentConvId = state.currentConversationId
      
      // 如果没有当前会话，自动创建一个
      if (!currentConvId) {
        currentConvId = `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        const newConversation: Conversation = {
          id: currentConvId,
          title: message.role === 'user' ? generateConversationTitle(message.content) : '新对话',
          messages: [newMessage],
          createdAt: Date.now(),
          updatedAt: Date.now(),
          mode: state.currentMode,
          selectedKB: state.selectedKB,
          currentImageContext: state.currentImageContext,
        }
        updatedConversations = [newConversation, ...updatedConversations]
      } else {
        // 更新当前会话的消息
        updatedConversations = state.conversations.map((conv) => {
          if (conv.id === currentConvId) {
            const updatedMessages = [...conv.messages, newMessage]
            
            // 如果是第一条用户消息，自动生成标题
            let updatedTitle = conv.title
            if (message.role === 'user' && conv.messages.length === 0) {
              updatedTitle = generateConversationTitle(message.content)
            }
            
            return {
              ...conv,
              messages: updatedMessages,
              title: updatedTitle,
              updatedAt: Date.now(),
            }
          }
          return conv
        })
      }
      
      // 保存到 localStorage
      saveConversationsToStorage(updatedConversations)
      
      // 获取更新后的消息列表
      const currentConv = updatedConversations.find(c => c.id === currentConvId)
      const updatedMessages = currentConv?.messages || []
      
      return {
        conversations: updatedConversations,
        currentConversationId: currentConvId,
        messages: updatedMessages,
      }
    }),
  
  clearMessages: () => {
    const state = get()
    if (state.currentConversationId) {
      // 清空当前会话的消息
      const updatedConversations = state.conversations.map((conv) => {
        if (conv.id === state.currentConversationId) {
          return {
            ...conv,
            messages: [],
            currentImageContext: [],
            updatedAt: Date.now(),
          }
        }
        return conv
      })
      saveConversationsToStorage(updatedConversations)
      set({
        conversations: updatedConversations,
        messages: [],
        currentImageContext: [],
        uploadedFiles: [],
      })
    } else {
      set({ messages: [], uploadedFiles: [] })
    }
  },
  
  setMode: (mode) => {
    const state = get()
    if (state.currentConversationId) {
      const updatedConversations = state.conversations.map((conv) => {
        if (conv.id === state.currentConversationId) {
          return { ...conv, mode, updatedAt: Date.now() }
        }
        return conv
      })
      saveConversationsToStorage(updatedConversations)
      set({
        conversations: updatedConversations,
        currentMode: mode,
      })
    } else {
      set({ currentMode: mode })
    }
  },
  
  setLoading: (loading) => set({ isLoading: loading }),
  
  setKnowledgeBases: (kbs) => set({ knowledgeBases: kbs }),
  
  setSelectedKB: (kb) => {
    const state = get()
    if (state.currentConversationId) {
      const updatedConversations = state.conversations.map((conv) => {
        if (conv.id === state.currentConversationId) {
          return { ...conv, selectedKB: kb, updatedAt: Date.now() }
        }
        return conv
      })
      saveConversationsToStorage(updatedConversations)
      set({
        conversations: updatedConversations,
        selectedKB: kb,
      })
    } else {
      set({ selectedKB: kb })
    }
  },
  
  addUploadedFile: (filename) =>
    set((state) => ({
      uploadedFiles: [...state.uploadedFiles, filename],
    })),
  
  clearUploadedFiles: () => set({ uploadedFiles: [] }),
  
  setEnableSearch: (enable) => set({ enableSearch: enable }),
  
  setSelectedModel: (model) => set({ selectedModel: model }),
  
  setCurrentImageContext: (images) => {
    const state = get()
    if (state.currentConversationId) {
      const updatedConversations = state.conversations.map((conv) => {
        if (conv.id === state.currentConversationId) {
          return { ...conv, currentImageContext: images, updatedAt: Date.now() }
        }
        return conv
      })
      saveConversationsToStorage(updatedConversations)
      set({
        conversations: updatedConversations,
        currentImageContext: images,
      })
    } else {
      set({ currentImageContext: images })
    }
  },
  
  clearImageContext: () => {
    const state = get()
    if (state.currentConversationId) {
      const updatedConversations = state.conversations.map((conv) => {
        if (conv.id === state.currentConversationId) {
          return { ...conv, currentImageContext: [], updatedAt: Date.now() }
        }
        return conv
      })
      saveConversationsToStorage(updatedConversations)
      set({
        conversations: updatedConversations,
        currentImageContext: [],
      })
    } else {
      set({ currentImageContext: [] })
    }
  },
  
  setInputValue: (value) => {
    set({ inputValueToSet: value })
  },
  
  // 会话管理 Actions
  createConversation: () => {
    const state = get()
    const newId = `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newConversation: Conversation = {
      id: newId,
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      mode: 'general',
      selectedKB: null,
      currentImageContext: [],
    }
    
    const updatedConversations = [newConversation, ...state.conversations]
    saveConversationsToStorage(updatedConversations)
    
    set({
      conversations: updatedConversations,
      currentConversationId: newId,
      messages: [],
      currentMode: 'general',
      selectedKB: null,
      currentImageContext: [],
      uploadedFiles: [],
    })
    
    return newId
  },
  
  switchConversation: (conversationId: string) => {
    const state = get()
    const conv = state.conversations.find(c => c.id === conversationId)
    if (conv) {
      set({
        currentConversationId: conversationId,
        messages: conv.messages,
        currentMode: conv.mode,
        selectedKB: conv.selectedKB || null,
        currentImageContext: conv.currentImageContext,
        uploadedFiles: [],
      })
    }
  },
  
  deleteConversation: (conversationId: string) => {
    const state = get()
    const updatedConversations = state.conversations.filter(c => c.id !== conversationId)
    saveConversationsToStorage(updatedConversations)
    
    // 如果删除的是当前会话，切换到第一个会话或创建新会话
    if (state.currentConversationId === conversationId) {
      if (updatedConversations.length > 0) {
        const firstConv = updatedConversations[0]
        set({
          conversations: updatedConversations,
          currentConversationId: firstConv.id,
          messages: firstConv.messages,
          currentMode: firstConv.mode,
          selectedKB: firstConv.selectedKB || null,
          currentImageContext: firstConv.currentImageContext,
          uploadedFiles: [],
        })
      } else {
        // 没有会话了，清空状态
        set({
          conversations: [],
          currentConversationId: null,
          messages: [],
          currentMode: 'general',
          selectedKB: null,
          currentImageContext: [],
          uploadedFiles: [],
        })
      }
    } else {
      set({ conversations: updatedConversations })
    }
  },
  
  updateConversationTitle: (conversationId: string, title: string) => {
    const state = get()
    const updatedConversations = state.conversations.map((conv) => {
      if (conv.id === conversationId) {
        return { ...conv, title, updatedAt: Date.now() }
      }
      return conv
    })
    saveConversationsToStorage(updatedConversations)
    set({ conversations: updatedConversations })
  },

  clearAllConversations: () => {
    // 清空所有对话
    const newConvId = `conv-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const newConversation: Conversation = {
      id: newConvId,
      title: '新对话',
      messages: [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
      mode: 'general',
      selectedKB: null,
      currentImageContext: [],
    }
    saveConversationsToStorage([newConversation])
    set({
      conversations: [newConversation],
      currentConversationId: newConvId,
      messages: [],
      currentMode: 'general',
      selectedKB: null,
      currentImageContext: [],
      uploadedFiles: [],
      enableSearch: false,
      selectedModel: null,
    })
  },
  }})
