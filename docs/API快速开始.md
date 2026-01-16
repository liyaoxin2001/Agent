# API 快速开始指南

本指南帮助你快速启动和使用 HuahuaChat API。

---

## 📋 前置条件

1. **Python 3.8+** 已安装
2. **依赖包** 已安装：
   ```bash
   pip install -r requirements.txt
   ```

3. **OpenAI API Key** 已配置（在 `.env` 文件中）

---

## 🚀 启动 API 服务

### 1. 配置环境变量

复制 `.env.example` 并重命名为 `.env`：

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 OpenAI API Key：

```env
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-ada-002
KB_ROOT_PATH=./data/knowledge_base
```

### 2. 启动 API 服务

**方式一：使用启动脚本（推荐）**

```bash
# 默认端口 8000
python run_api.py

# 自定义端口
python run_api.py --port 8080

# 开发模式（自动重载）
python run_api.py --reload
```

**方式二：直接使用 uvicorn**

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

### 3. 验证服务启动

启动成功后，你应该看到类似的输出：

```
======================================================================
🚀 HuahuaChat API 正在启动...
======================================================================
📦 ServiceContainer 初始化完成
✅ KnowledgeBaseManager 初始化成功，根目录: ./data/knowledge_base
✅ LLM 初始化成功，模型: gpt-3.5-turbo
✅ Embedding 初始化成功，模型: text-embedding-ada-002
✅ 所有服务初始化成功
======================================================================
📖 API 文档: http://localhost:8000/docs
📖 ReDoc 文档: http://localhost:8000/redoc
======================================================================
```

访问 http://localhost:8000/docs 查看 API 文档。

---

## 📚 API 端点概览

### 健康检查

```bash
GET /health
```

**示例**：

```bash
curl http://localhost:8000/health
```

### 知识库管理

#### 创建知识库

```bash
POST /api/v1/kb/create
```

**请求体**：

```json
{
  "name": "my_kb",
  "description": "我的知识库",
  "embedding_model": "text-embedding-ada-002"
}
```

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/kb/create \
  -H "Content-Type: application/json" \
  -d '{"name": "my_kb", "description": "我的知识库"}'
```

#### 列出知识库

```bash
GET /api/v1/kb/list
```

**示例**：

```bash
curl http://localhost:8000/api/v1/kb/list
```

#### 获取知识库详情

```bash
GET /api/v1/kb/{kb_name}
```

**示例**：

```bash
curl http://localhost:8000/api/v1/kb/my_kb
```

#### 删除知识库

```bash
DELETE /api/v1/kb/{kb_name}
```

**示例**：

```bash
curl -X DELETE http://localhost:8000/api/v1/kb/my_kb
```

### 文档管理

#### 上传文档

```bash
POST /api/v1/kb/{kb_name}/upload
```

**支持的文件类型**：`.txt`, `.pdf`, `.md`, `.markdown`

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/kb/my_kb/upload \
  -F "file=@path/to/document.txt"
```

#### 列出文档

```bash
GET /api/v1/kb/{kb_name}/documents
```

**示例**：

```bash
curl http://localhost:8000/api/v1/kb/my_kb/documents
```

#### 删除文档

```bash
DELETE /api/v1/kb/{kb_name}/documents/{filename}
```

**示例**：

```bash
curl -X DELETE http://localhost:8000/api/v1/kb/my_kb/documents/document.txt
```

### 聊天问答

#### 发送消息

```bash
POST /api/v1/chat
```

**请求体**：

```json
{
  "question": "Python 是什么？",
  "kb_name": "my_kb",
  "conversation_id": "conv-123",
  "stream": false,
  "max_steps": 5
}
```

**示例**：

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Python 是什么？",
    "kb_name": "my_kb"
  }'
```

**响应**：

```json
{
  "answer": "Python 是一种高级编程语言...",
  "conversation_id": "conv-abc123",
  "retrieved_docs": [
    {
      "content": "Python 相关内容...",
      "source": "python_intro.txt",
      "score": 0.95
    }
  ],
  "metadata": {
    "retrieval_score": 0.91,
    "confidence_score": 0.88,
    "step_count": 2,
    "kb_name": "my_kb"
  }
}
```

---

## 🧪 运行测试

我们提供了完整的 API 测试脚本。

### 1. 启动 API 服务

```bash
python run_api.py
```

### 2. 运行测试（在新终端）

```bash
python examples/test_api.py
```

测试将自动：
1. ✅ 检查健康状态
2. ✅ 创建测试知识库
3. ✅ 列出知识库
4. ✅ 上传测试文档
5. ✅ 列出文档
6. ✅ 执行聊天问答
7. ✅ （可选）删除测试知识库

---

## 📖 完整工作流示例

以下是一个完整的使用示例：

### 1. 创建知识库

```bash
curl -X POST http://localhost:8000/api/v1/kb/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "tech_kb",
    "description": "技术知识库"
  }'
```

### 2. 上传文档

```bash
# 上传 Python 教程
curl -X POST http://localhost:8000/api/v1/kb/tech_kb/upload \
  -F "file=@python_tutorial.txt"

# 上传 Java 教程
curl -X POST http://localhost:8000/api/v1/kb/tech_kb/upload \
  -F "file=@java_tutorial.pdf"
```

### 3. 查看文档

```bash
curl http://localhost:8000/api/v1/kb/tech_kb/documents
```

### 4. 提问

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Python 和 Java 有什么区别？",
    "kb_name": "tech_kb"
  }'
```

### 5. 多轮对话

```bash
# 第一轮
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Python 是什么？",
    "kb_name": "tech_kb"
  }'

# 使用返回的 conversation_id 进行第二轮
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "它的主要特点是什么？",
    "kb_name": "tech_kb",
    "conversation_id": "conv-abc123"
  }'
```

---

## 🔧 常见问题

### 1. API 启动失败

**问题**：服务无法启动

**解决**：
- 检查 `.env` 文件是否配置正确
- 确认 OpenAI API Key 有效
- 查看错误日志

### 2. 知识库创建失败

**问题**：创建知识库返回 500 错误

**解决**：
- 确认 `KB_ROOT_PATH` 目录存在且可写
- 检查 Embedding 模型配置

### 3. 聊天返回空答案

**问题**：聊天返回 "抱歉，我无法回答这个问题"

**解决**：
- 确认知识库中有相关文档
- 检查文档是否成功上传
- 查看检索分数（retrieval_score）

### 4. 文档上传失败

**问题**：上传文档返回 400 或 500 错误

**解决**：
- 确认文件类型支持（.txt, .pdf, .md）
- 检查文件大小（不宜过大）
- 查看服务器日志

---

## 📊 API 文档

访问以下 URL 查看交互式 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🛠️ 高级配置

### 自定义端口和主机

```bash
python run_api.py --host 127.0.0.1 --port 8080
```

### 多进程模式（生产环境）

```bash
python run_api.py --workers 4
```

### 开发模式（自动重载）

```bash
python run_api.py --reload
```

---

## 📝 下一步

- 阅读 [架构设计](./架构设计.md) 了解系统设计
- 查看 [开发指南](./开发指南.md) 学习如何扩展功能
- 尝试 [Streamlit 前端](./Streamlit前端指南.md)（阶段四）

---

**Happy Coding! 🚀**
