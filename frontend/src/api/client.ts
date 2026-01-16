import axios from 'axios'

const API_BASE_URL = '/api/v1'

export interface ChatRequest {
  question: string
  kb_name?: string
  conversation_id?: string
  stream?: boolean
  max_steps?: number
}

export interface ChatResponse {
  answer: string
  conversation_id?: string
  sources?: any[]
}

export interface KnowledgeBase {
  name: string
  description?: string
  document_count: number
}

// Chat API
export const chatAPI = {
  // RAG 模式聊天
  chat: async (request: ChatRequest): Promise<ChatResponse> => {
    const response = await axios.post(`${API_BASE_URL}/chat`, request)
    return response.data
  },
  
  // 通用模式聊天（直接调用 OpenAI，支持图片和模型选择）
  chatGeneral: async (
    question: string, 
    images?: string[], 
    modelName?: string | null,
    conversationHistory?: Array<{role: string, content: string, images?: string[]}>
  ): Promise<string> => {
    const response = await axios.post(`${API_BASE_URL}/chat/general`, {
      question: question,
      images: images || [], // 使用images字段，传递base64数据
      model_name: modelName || null,
      conversation_history: conversationHistory || [], // 传递对话历史
    })
    return response.data.answer
  },
  
  // 搜索模式
  searchAndChat: async (question: string): Promise<{ answer: string; sources: any[] }> => {
    const response = await axios.post(`${API_BASE_URL}/chat/search`, {
      question: question,
    })
    return {
      answer: response.data.answer,
      sources: response.data.retrieved_docs || [],
    }
  },
  
  // 获取默认模型信息
  getDefaultModel: async () => {
    const response = await axios.get(`${API_BASE_URL}/chat/default-model`)
    return response.data
  },
}

// Knowledge Base API
export const kbAPI = {
  // 获取知识库列表
  list: async (): Promise<{ knowledge_bases: KnowledgeBase[] }> => {
    const response = await axios.get(`${API_BASE_URL}/kb/list`)
    return response.data
  },
  
  // 创建知识库
  create: async (name: string, description?: string) => {
    const response = await axios.post(`${API_BASE_URL}/kb/create`, { name, description })
    return response.data
  },
  
  // 上传文件到知识库
  upload: async (kbName: string, file: File) => {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await axios.post(`${API_BASE_URL}/kb/${kbName}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },
  
  // 删除知识库
  delete: async (kbName: string) => {
    const response = await axios.delete(`${API_BASE_URL}/kb/${kbName}`)
    return response.data
  },
  
  // 获取知识库文档列表
  listDocuments: async (kbName: string) => {
    const response = await axios.get(`${API_BASE_URL}/kb/${kbName}/documents`)
    return response.data
  },
  
  // 删除文档
  deleteDocument: async (kbName: string, filename: string) => {
    const response = await axios.delete(`${API_BASE_URL}/kb/${kbName}/documents/${filename}`)
    return response.data
  },
}

// Health check
export const healthCheck = async (): Promise<boolean> => {
  try {
    const response = await axios.get('/health')
    return response.status === 200
  } catch {
    return false
  }
}
