# 🐱 HuahuaChat - 智能对话助手

一个基于 LangChain 和 LangGraph 构建的现代化 RAG（检索增强生成）聊天应用，支持知识库问答、通用对话、联网搜索和图片理解功能。

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ 功能特性

### 🎯 核心功能

- **📚 知识库问答**：上传文档创建知识库，基于文档内容精准回答
- **💬 通用对话**：像 ChatGPT 一样进行自然对话
- **🔍 联网搜索**：实时搜索互联网信息并智能总结
- **🖼️ 图片理解**：支持上传图片，AI 可以理解和分析图片内容
- **🎨 多会话管理**：支持创建、切换、删除多个对话会话
- **🌓 主题切换**：支持亮色/暗色模式切换
- **🤖 模型选择**：支持动态切换不同的 AI 模型

### 🎨 界面特性

- **现代化 UI**：仿 ChatGPT 风格的 React 前端界面
- **响应式设计**：适配不同屏幕尺寸
- **流畅交互**：实时消息流、自动滚动、输入聚焦
- **代码高亮**：支持 Markdown 和代码语法高亮

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Node.js 16+ (用于前端)
- OpenAI API Key (或其他兼容的 API)

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd HuahuaChat
```

#### 2. 后端设置

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 3. 配置环境变量

创建 `.env` 文件（参考 `env_template.txt`）：

```env
# OpenAI API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，用于代理
OPENAI_MODEL=gpt-3.5-turbo  # 默认模型

# 可选：其他配置
# EMBEDDING_MODEL=text-embedding-3-small
```

#### 4. 前端设置

```bash
cd frontend

# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

#### 5. 启动后端 API

```bash
# 在项目根目录
python run_api.py

# 或使用 uvicorn 直接启动
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

#### 6. 访问应用

- 前端：http://localhost:3000
- API 文档：http://localhost:8000/docs

## 📁 项目结构

```
HuahuaChat/
├── frontend/              # React 前端应用
│   ├── src/
│   │   ├── components/   # React 组件
│   │   ├── api/          # API 客户端
│   │   ├── store/        # 状态管理 (Zustand)
│   │   └── hooks/        # 自定义 Hooks
│   ├── public/           # 静态资源（包含 logo.png）
│   └── package.json
│
├── src/                  # Python 后端代码
│   ├── api/              # FastAPI 路由和模型
│   │   ├── routers/      # API 路由
│   │   └── models.py     # Pydantic 模型
│   ├── core/             # 核心功能模块
│   │   ├── llm/          # LLM 封装
│   │   ├── embedding/    # Embedding 模型
│   │   ├── vectorstore/  # 向量数据库
│   │   ├── document/     # 文档处理
│   │   └── chain/        # RAG Chain
│   ├── agent/            # LangGraph Agent
│   │   ├── state.py      # Agent 状态定义
│   │   ├── nodes.py      # Agent 节点
│   │   └── graph.py      # Agent 图定义
│   └── knowledge_base/   # 知识库管理
│
├── data/                 # 数据目录
│   └── knowledge_base/   # 知识库存储
│
├── examples/             # 示例代码
├── docs/                 # 文档和学习笔记
├── tests/                # 测试文件
├── run_api.py           # API 启动脚本
├── requirements.txt      # Python 依赖
└── README.md            # 本文件
```

## 🎯 使用指南

### 知识库问答

1. 点击左侧「知识库问答」模式
2. 点击「创建知识库」按钮
3. 输入知识库名称和描述
4. 上传文档（支持 txt, pdf, md, docx）
5. 等待文档处理完成
6. 开始提问，AI 会基于文档内容回答

### 通用对话

1. 点击左侧「通用对话」模式
2. 选择模型（可选）
3. 直接开始对话
4. 支持上传图片进行视觉理解

### 联网搜索

1. 在「通用对话」模式下
2. 点击输入框上方的搜索图标启用联网搜索
3. 输入问题，AI 会搜索并总结信息

### 多会话管理

- 点击「新对话」创建新会话
- 左侧显示所有会话列表
- 点击会话切换，点击删除按钮删除会话
- 所有会话自动保存到本地

## 🔧 配置说明

### 模型配置

在 `.env` 文件中设置：

```env
OPENAI_MODEL=gpt-4o  # 推荐使用支持视觉的模型
```

支持的视觉模型：
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4-turbo`
- `gpt-4-vision-preview`

### Logo 配置

将您的 logo 图片（建议 128x128 像素）命名为 `logo.png` 并放置在 `frontend/public/` 目录下。

## 📚 技术栈

### 后端

- **FastAPI** - 现代 Python Web 框架
- **LangChain** - LLM 应用开发框架
- **LangGraph** - 状态机和工作流框架
- **FAISS** - 向量相似度搜索
- **Pydantic** - 数据验证

### 前端

- **React 18** - UI 框架
- **TypeScript** - 类型安全
- **Vite** - 构建工具
- **Tailwind CSS** - 样式框架
- **Zustand** - 状态管理
- **Axios** - HTTP 客户端

## 🛠️ 开发

### 运行测试

```bash
# 运行 Python 测试
pytest tests/

# 运行前端测试
cd frontend
npm test
```

### 代码格式化

```bash
# Python
black src/

# TypeScript
cd frontend
npm run lint
```

## 📖 API 文档

启动后端后，访问 http://localhost:8000/docs 查看完整的 API 文档。

主要 API 端点：

- `POST /api/v1/chat` - 知识库问答
- `POST /api/v1/chat/general` - 通用对话
- `POST /api/v1/chat/search` - 联网搜索
- `GET /api/v1/knowledge-bases` - 获取知识库列表
- `POST /api/v1/knowledge-bases/{kb_name}/documents` - 上传文档

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📝 许可证

MIT License

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM 应用框架
- [LangGraph](https://github.com/langchain-ai/langgraph) - 状态机框架
- [FastAPI](https://fastapi.tiangolo.com/) - Web 框架
- [React](https://react.dev/) - UI 框架

## 📧 联系方式

如有问题或建议，请提交 Issue。

---

**HuahuaChat** - 让 AI 对话更简单 🐱
