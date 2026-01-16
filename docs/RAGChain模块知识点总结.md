# RAG Chain 模块知识点总结

## 目录

1. [模块概述](#一模块概述)
2. [核心概念](#二核心概念---rag-的本质)
3. [类设计与架构](#三类设计与架构)
4. [RAGChain 类详解](#四ragchain-类详解)
5. [ConversationalRAGChain 类详解](#五conversationalragchain-类详解)
6. [技术细节与实现原理](#六技术细节与实现原理)
7. [设计模式与最佳实践](#七设计模式与最佳实践)
8. [常见问题与优化](#八常见问题与优化)
9. [完整使用示例](#九完整使用示例)

---

## 一、模块概述

### 1.1 RAG Chain 是什么？

**RAG Chain** 是 **检索增强生成（Retrieval-Augmented Generation）** 系统的核心组件，负责协调**向量检索**和**语言模型生成**两个关键步骤。

```
传统 LLM 问答：
用户问题 → LLM → 答案
问题：可能不准确、过时、编造信息

RAG 问答：
用户问题 → 检索相关文档 → 组装 Prompt → LLM → 准确答案
           ↑
      向量数据库
```

### 1.2 模块的作用

| 作用 | 说明 |
|------|------|
| **信息检索** | 从知识库中找到与问题相关的文档 |
| **上下文组装** | 将检索到的文档格式化为结构化上下文 |
| **Prompt 工程** | 使用模板将上下文和问题组合成完整 Prompt |
| **答案生成** | 调用 LLM 基于上下文生成准确答案 |
| **历史管理** | 维护多轮对话的上下文（ConversationalRAGChain） |

### 1.3 应用场景

1. **企业知识库问答**
   - 员工查询公司政策、流程
   - 技术文档检索与问答

2. **智能客服系统**
   - 基于产品手册回答客户问题
   - 自动化客户支持

3. **技术文档助手**
   - 代码库文档问答
   - API 使用指南

4. **教育培训**
   - 课程内容问答
   - 学习资料检索

---

## 二、核心概念 - RAG 的本质

### 2.1 什么是 RAG？

**RAG = Retrieval（检索） + Augmented（增强） + Generation（生成）**

```python
# RAG 的核心思想
def rag_process(question):
    # 1. Retrieval - 检索
    relevant_docs = search_in_knowledge_base(question)
    
    # 2. Augmented - 增强
    context = format_documents(relevant_docs)
    prompt = f"基于上下文: {context}\n问题: {question}"
    
    # 3. Generation - 生成
    answer = llm.generate(prompt)
    
    return answer
```

### 2.2 为什么需要 RAG？

#### 传统 LLM 的局限

| 问题 | 说明 | RAG 解决方案 |
|------|------|-------------|
| **知识过时** | 训练数据固定，无法更新 | ✅ 更新知识库即可 |
| **缺乏专业知识** | 通用模型缺乏领域深度 | ✅ 导入专业文档 |
| **信息幻觉** | 容易编造不存在的信息 | ✅ 基于真实文档回答 |
| **无法追溯** | 无法提供信息来源 | ✅ 可引用文档编号 |
| **token 限制** | 无法处理大量信息 | ✅ 只检索相关部分 |

#### RAG 的价值

```python
# 示例：技术支持场景
question = "如何重置密码？"

# 传统 LLM（可能不准确）
answer_traditional = llm.generate("如何重置密码？")
# 可能返回：通用的密码重置步骤，不针对具体系统

# RAG（准确且可靠）
# 1. 从系统文档中检索相关内容
docs = vectorstore.search("如何重置密码？")
# 找到：用户手册第 3.2 节"密码重置流程"

# 2. 基于真实文档生成答案
context = format_docs(docs)
answer_rag = llm.generate(f"基于{context}，回答：{question}")
# 返回：具体的、准确的系统密码重置步骤
```

### 2.3 RAG 的工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 完整工作流程                          │
└─────────────────────────────────────────────────────────────┘

步骤 1: 用户提问
   └─> "Python 是什么时候发布的？"

步骤 2: 向量化查询
   └─> Embedding Model
       └─> [0.1, 0.3, -0.2, ..., 0.5]  # 查询向量

步骤 3: 向量检索
   └─> VectorStore.similarity_search()
       └─> 计算相似度，返回 Top-K 文档
           └─> [Document1, Document2, Document3]

步骤 4: 上下文组装
   └─> "[文档1]\nPython 于 1991 年发布\n\n[文档2]\n..."

步骤 5: Prompt 组装
   └─> """
       上下文：[文档1] Python 于 1991 年发布...
       问题：Python 是什么时候发布的？
       请基于上下文回答。
       """

步骤 6: LLM 生成
   └─> LLM.generate(prompt)
       └─> "根据上下文，Python 于 1991 年发布。"

步骤 7: 返回答案
   └─> 用户看到准确的回答
```

---

## 三、类设计与架构

### 3.1 模块结构

```
src/core/chain/rag_chain.py
├── RAGChain                      # 基础 RAG 类
│   ├── __init__()               # 初始化
│   ├── _get_default_template()  # 获取 Prompt 模板
│   ├── query()                  # 执行 RAG 查询
│   └── stream_query()           # 流式 RAG 查询
│
└── ConversationalRAGChain        # 对话式 RAG 类（继承 RAGChain）
    ├── __init__()               # 初始化（添加历史存储）
    ├── _get_default_template()  # 带历史的模板
    ├── query()                  # 带历史的查询
    ├── stream_query()           # 带历史的流式查询
    ├── _format_history()        # 格式化历史
    ├── _add_to_history()        # 添加到历史
    └── clear_history()          # 清空历史
```

### 3.2 类关系图

```
┌─────────────────┐
│   BaseLLM       │  (抽象接口)
└─────────────────┘
         ▲
         │ 使用
         │
┌─────────────────┐         ┌──────────────────────┐
│  RAGChain       │◄────────│ BaseVectorStore      │
│                 │  使用    │                      │
│  - llm          │         │  (抽象接口)           │
│  - vectorstore  │         └──────────────────────┘
│  - prompt       │
└─────────────────┘
         ▲
         │ 继承
         │
┌─────────────────────────┐
│ ConversationalRAGChain  │
│                         │
│  + chat_history        │  (新增属性)
│  + max_history         │
└─────────────────────────┘
```

### 3.3 设计原则

#### 1. 单一职责原则（SRP）

```python
# RAGChain 只负责 RAG 流程，不负责：
# ❌ LLM 的具体实现 → 由 BaseLLM 负责
# ❌ 向量存储的实现 → 由 BaseVectorStore 负责
# ❌ Embedding 计算 → 由 BaseEmbedding 负责

class RAGChain:
    def query(self, question):
        # ✅ 只负责协调各个组件
        docs = self.vectorstore.similarity_search(question)
        context = self._format_context(docs)
        prompt = self._assemble_prompt(context, question)
        answer = self.llm.generate(prompt)
        return answer
```

#### 2. 开闭原则（OCP）

```python
# 对扩展开放
class ConversationalRAGChain(RAGChain):
    # ✅ 通过继承扩展功能，不修改父类
    def query(self, question):
        # 添加历史处理逻辑
        history = self._format_history()
        # ... 其余逻辑
        
# 对修改封闭
# ❌ 不需要修改 RAGChain 的代码
# ✅ 父类和子类可以共存，互不影响
```

#### 3. 依赖倒置原则（DIP）

```python
# 依赖抽象接口，不依赖具体实现
class RAGChain:
    def __init__(
        self,
        llm: BaseLLM,              # ✅ 依赖抽象
        vectorstore: BaseVectorStore  # ✅ 依赖抽象
    ):
        self.llm = llm
        self.vectorstore = vectorstore

# 可以传入任何实现了接口的类
rag1 = RAGChain(llm=OpenAILLM(...), vectorstore=FAISSVectorStore(...))
rag2 = RAGChain(llm=OllamaLLM(...), vectorstore=ChromaVectorStore(...))
```

---

## 四、RAGChain 类详解

### 4.1 类的职责

**RAGChain** 是基础的 RAG 实现，负责核心的检索-生成流程。

```python
class RAGChain:
    """
    核心职责：
    1. 管理 LLM 和 VectorStore 实例
    2. 提供默认 Prompt 模板
    3. 实现 RAG 查询流程
    4. 支持流式和非流式生成
    """
```

### 4.2 初始化方法 `__init__()`

#### 方法签名

```python
def __init__(
    self,
    llm: BaseLLM,                        # LLM 实例
    vectorstore: BaseVectorStore,        # 向量存储实例
    prompt_template: Optional[str] = None  # 可选的自定义模板
):
```

#### 参数详解

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `llm` | BaseLLM | ✅ | LLM 实例，用于生成答案 |
| `vectorstore` | BaseVectorStore | ✅ | 向量存储实例，用于检索文档 |
| `prompt_template` | Optional[str] | ❌ | 自定义 Prompt 模板，不提供则使用默认 |

#### 工作流程

```python
def __init__(self, llm, vectorstore, prompt_template=None):
    # 步骤 1: 保存组件引用
    self.llm = llm
    self.vectorstore = vectorstore
    
    # 步骤 2: 设置 Prompt 模板
    # 逻辑：prompt_template or self._get_default_template()
    # 如果用户提供了模板（不是 None），使用用户的
    # 否则调用 _get_default_template() 获取默认模板
    self.prompt_template = prompt_template or self._get_default_template()
```

#### Python 知识点：`or` 运算符

```python
# x or y 的逻辑
# - 如果 x 为真（True, 非空字符串, 非 None），返回 x
# - 如果 x 为假（False, None, 空字符串），返回 y

# 示例 1
value = "用户提供" or "默认值"  # 返回 "用户提供"

# 示例 2
value = None or "默认值"  # 返回 "默认值"

# 在 RAGChain 中
self.prompt_template = prompt_template or self._get_default_template()
# 如果 prompt_template 是 None，调用 _get_default_template()
```

### 4.3 默认模板方法 `_get_default_template()`

#### 为什么需要默认模板？

```python
# 如果没有默认模板
rag = RAGChain(llm, vectorstore)  # 必须提供 prompt_template
# ❌ 用户必须自己设计 Prompt，学习成本高

# 有默认模板
rag = RAGChain(llm, vectorstore)  # ✅ 自动使用企业级模板
# ✅ 开箱即用，无需 Prompt 工程知识
```

#### 企业级模板设计

```python
template = """你是一位专业的智能助手，擅长基于提供的上下文信息进行精准、客观的回答。

## 上下文信息
{context}

## 用户问题
{question}

## 回答指南
请严格遵循以下要求提供高质量的回答：

1. **准确性优先**：仅基于上述上下文信息回答，确保信息准确可靠
2. **完整性保证**：如果上下文中包含相关信息，请提供详细、全面的回答
3. **诚实原则**：如果上下文中没有足够信息回答问题，请明确说明"根据提供的信息，我无法完整回答这个问题"
4. **禁止臆测**：严禁编造、推测或添加上下文中不存在的信息
5. **结构化表达**：使用清晰的逻辑结构组织回答，必要时使用要点列举
6. **简洁专业**：语言简洁明了，避免冗余，保持专业性

## 你的回答
"""
```

#### 模板设计要点

| 要素 | 作用 | 示例 |
|------|------|------|
| **角色定位** | 告诉 LLM 它的身份 | "你是一位专业的智能助手" |
| **结构化** | 使用 Markdown 标题 | `##` 分隔各部分 |
| **占位符** | 动态填充内容 | `{context}`, `{question}` |
| **回答指南** | 明确的指令 | 6 条详细要求 |
| **边界处理** | 处理边界情况 | "如果上下文中没有..." |

#### 为什么一个模板就够？

**关键理解**：模板定义"如何回答"，不是"回答什么"

```python
# 同一个模板，处理不同问题
template = "上下文：{context}\n问题：{question}\n回答："

# 问题 1：技术
context1 = "Python 是编程语言"
question1 = "什么是 Python？"
prompt1 = template.format(context=context1, question=question1)

# 问题 2：历史
context2 = "1969 年互联网诞生"
question2 = "互联网何时诞生？"
prompt2 = template.format(context=context2, question=question2)

# 所有问题用同一个模板！内容通过占位符动态填充
```

### 4.4 核心查询方法 `query()`

#### 方法签名

```python
def query(self, question: str, k: int = 4) -> str:
```

#### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `question` | str | 必填 | 用户的问题文本 |
| `k` | int | 4 | 检索的文档数量，建议 3-10 |

**k 值选择指南**：

```python
# k 太小（k=1）
# ❌ 可能遗漏重要信息
# ❌ 容易受噪声影响
relevant_docs = vectorstore.search(question, k=1)
# 只有 1 个文档，信息可能不全面

# k 适中（k=3-5）
# ✅ 信息较全面
# ✅ token 消耗合理
relevant_docs = vectorstore.search(question, k=4)
# 平衡信息量和性能

# k 太大（k=20）
# ❌ 引入大量噪声
# ❌ token 消耗高
# ❌ 可能降低准确性
relevant_docs = vectorstore.search(question, k=20)
# 太多不相关文档干扰 LLM
```

#### 完整工作流程

```python
def query(self, question: str, k: int = 4) -> str:
    try:
        # ===== 步骤 1: 向量检索 =====
        relevant_docs = self.vectorstore.similarity_search(question, k=k)
        
        # ===== 步骤 2: 边界检查 =====
        if not relevant_docs:
            return "抱歉，在知识库中没有找到相关信息。"
        
        # ===== 步骤 3: 上下文组装 =====
        contexts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_piece = f"[文档{i}]\n{doc.page_content}"
            contexts.append(context_piece)
        context = "\n\n".join(contexts)
        
        # ===== 步骤 4: Prompt 组装 =====
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # ===== 步骤 5: LLM 生成 =====
        answer = self.llm.generate(prompt)
        
        # ===== 步骤 6: 返回答案 =====
        return answer
        
    except Exception as e:
        raise Exception(f"RAG查询失败：{str(e)}") from e
```

#### 步骤详解

**步骤 1: 向量检索**

```python
relevant_docs = self.vectorstore.similarity_search(question, k=k)

# 底层原理：
# 1. 将 question 转换为向量 q_vec
# 2. 计算 q_vec 与所有文档向量的相似度
# 3. 返回相似度最高的 k 个文档

# 时间复杂度：
# - 精确搜索：O(n)，n 是文档数量
# - 近似搜索（FAISS）：O(log n)
```

**步骤 2: 边界检查**

```python
if not relevant_docs:
    return "抱歉，在知识库中没有找到相关信息。"

# 为什么需要？
# 1. 向量库可能为空
# 2. 查询与所有文档都不相关（相似度太低）
# 3. 提供友好的用户体验
```

**步骤 3: 上下文组装**

```python
contexts = []
for i, doc in enumerate(relevant_docs, 1):
    context_piece = f"[文档{i}]\n{doc.page_content}"
    contexts.append(context_piece)
context = "\n\n".join(contexts)

# 为什么添加文档编号？
# 1. 便于追溯信息来源
# 2. LLM 可以引用"根据文档 2..."
# 3. 提高答案可信度

# 为什么用双换行？
# 1. 在文档之间创建明显的视觉分隔
# 2. 帮助 LLM 理解文档边界
# 3. 提高 Prompt 可读性
```

**步骤 4: Prompt 组装**

```python
prompt = self.prompt_template.format(
    context=context,
    question=question
)

# format() 方法做了什么？
template = "上下文：{context}\n问题：{question}"
prompt = template.format(context="文档内容", question="用户问题")
# 结果：
# 上下文：文档内容
# 问题：用户问题
```

**步骤 5: LLM 生成**

```python
answer = self.llm.generate(prompt)

# LLM 做了什么？
# 1. 理解 Prompt 中的指令
# 2. 分析上下文信息
# 3. 基于上下文回答问题
# 4. 返回生成的文本
```

#### Python 知识点

**enumerate() 函数**：

```python
# enumerate(iterable, start=0) 返回 (索引, 元素)
items = ["a", "b", "c"]

# 从 0 开始
for i, item in enumerate(items):
    print(f"{i}: {item}")
# 输出: 0: a, 1: b, 2: c

# 从 1 开始
for i, item in enumerate(items, 1):
    print(f"{i}: {item}")
# 输出: 1: a, 2: b, 3: c
```

**join() 方法**：

```python
# str.join(list) 用指定字符串连接列表元素
parts = ["part1", "part2", "part3"]

" ".join(parts)       # "part1 part2 part3"
"\n".join(parts)      # part1\npart2\npart3
"\n\n".join(parts)    # part1\n\npart2\n\npart3
```

**异常处理**：

```python
try:
    # 可能出错的代码
    result = risky_operation()
except Exception as e:
    # 捕获异常
    # str(e): 获取错误信息
    # from e: 保留原始堆栈信息
    raise Exception(f"操作失败: {str(e)}") from e
```

### 4.5 流式查询方法 `stream_query()`

#### query() vs stream_query()

| 特性 | query() | stream_query() |
|------|---------|----------------|
| **返回方式** | 一次性返回完整答案 | 逐个产出文本片段 |
| **用户体验** | 需要等待 | 实时看到生成过程 |
| **首字延迟** | 高（需等待完整生成） | 低（立即返回第一个片段） |
| **适用场景** | 短答案、批量处理 | 长答案、交互式对话 |
| **内存占用** | 完整答案在内存中 | 流式产出，内存友好 |

#### 实现原理

```python
def stream_query(self, question: str, k: int = 4):
    try:
        # 步骤 1-3: 检索和组装（与 query() 相同）
        relevant_docs = self.vectorstore.similarity_search(question, k=k)
        if not relevant_docs:
            yield "抱歉，我在知识库中没有找到相关信息。"
            return
        
        contexts = [
            f"[文档{i}]\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs, 1)
        ]
        context = "\n\n".join(contexts)
        prompt = self.prompt_template.format(context=context, question=question)
        
        # 步骤 4: 流式生成（关键区别）
        for chunk in self.llm.stream_generate(prompt):
            yield chunk  # 逐个产出文本片段
            
    except Exception as e:
        raise Exception(f"流式RAG查询失败：{str(e)}") from e
```

#### Python 知识点：生成器（Generator）

**什么是生成器？**

```python
# 普通函数 - 使用 return
def normal_function():
    return [1, 2, 3]  # 一次性返回整个列表

result = normal_function()
print(result)  # [1, 2, 3]

# 生成器函数 - 使用 yield
def generator_function():
    yield 1  # 产出第一个值，暂停
    yield 2  # 产出第二个值，暂停
    yield 3  # 产出第三个值

for value in generator_function():
    print(value)  # 逐个输出: 1, 2, 3
```

**yield 的工作原理**：

```python
def my_generator():
    print("开始")
    yield "第一个值"  # 暂停在这里，返回值
    print("继续")
    yield "第二个值"  # 暂停在这里，返回值
    print("结束")

# 使用生成器
gen = my_generator()
print(next(gen))  # 输出: 开始, 第一个值
print(next(gen))  # 输出: 继续, 第二个值
```

**在 RAG 中的应用**：

```python
# 流式生成的工作流程
def stream_query(self, question, k=4):
    # ... 检索和组装 ...
    
    # LLM 流式生成
    for chunk in self.llm.stream_generate(prompt):
        # chunk 是文本片段，例如 "Python", " 是", "一种"...
        yield chunk  # 立即返回给调用者
    
# 使用
for chunk in rag_chain.stream_query("什么是 Python？"):
    print(chunk, end="", flush=True)  # 打字机效果
```

**列表推导式**：

```python
# 传统方式
contexts = []
for i, doc in enumerate(relevant_docs, 1):
    contexts.append(f"[文档{i}]\n{doc.page_content}")

# 列表推导式（更简洁）
contexts = [
    f"[文档{i}]\n{doc.page_content}"
    for i, doc in enumerate(relevant_docs, 1)
]

# 语法：[表达式 for 变量 in 可迭代对象]
```

---

## 五、ConversationalRAGChain 类详解

### 5.1 为什么需要对话历史？

#### 问题场景

```python
# 没有历史的对话
rag = RAGChain(llm, vectorstore)

# 第一轮
answer1 = rag.query("Python 是什么？")
# 答案：Python 是一种编程语言

# 第二轮
answer2 = rag.query("它有什么特点？")  # ❌ "它" 指什么？
# 答案：无法理解 "它" 指代什么
```

```python
# 有历史的对话
conv_rag = ConversationalRAGChain(llm, vectorstore)

# 第一轮
answer1 = conv_rag.query("Python 是什么？")
# 答案：Python 是一种编程语言
# 历史：[("Python 是什么？", "Python 是一种编程语言")]

# 第二轮
answer2 = conv_rag.query("它有什么特点？")  # ✅ 能理解 "它" = Python
# 答案：Python 的特点包括简洁易学...
# 历史：[("Python 是什么？", "..."), ("它有什么特点？", "...")]
```

#### 核心差异

| 特性 | RAGChain | ConversationalRAGChain |
|------|----------|----------------------|
| **状态** | 无状态（每次查询独立） | 有状态（维护历史） |
| **上下文** | 只有当前问题 | 当前问题 + 历史对话 |
| **指代消解** | ❌ 无法理解代词 | ✅ 基于历史理解 |
| **连续追问** | ❌ 不支持 | ✅ 支持 |
| **内存占用** | 低 | 中（存储历史） |

### 5.2 继承关系与设计

```python
class ConversationalRAGChain(RAGChain):
    """
    继承 RAGChain，复用基础功能
    扩展：添加对话历史管理
    """
```

**继承的好处**：

```python
# 1. 代码复用
# ConversationalRAGChain 自动获得父类的所有方法
conv_rag = ConversationalRAGChain(llm, vectorstore)
conv_rag._get_default_template()  # ✅ 可以调用（如果没重写）

# 2. 多态性
# 可以用父类类型引用子类对象
def process_rag(rag_chain: RAGChain):
    return rag_chain.query("问题")

# ✅ 两种类型都可以传入
process_rag(RAGChain(llm, vectorstore))
process_rag(ConversationalRAGChain(llm, vectorstore))

# 3. 扩展性
# 不修改父类，通过重写方法扩展功能
class ConversationalRAGChain(RAGChain):
    def query(self, question):
        # 添加历史处理逻辑
        history = self._format_history()
        # ... 其余与父类类似
```

### 5.3 初始化方法

```python
def __init__(
    self,
    llm: BaseLLM,
    vectorstore: BaseVectorStore,
    prompt_template: Optional[str] = None,
    max_history: int = 100  # 新增参数
):
    # 调用父类初始化
    super().__init__(llm, vectorstore, prompt_template)
    
    # 新增属性
    self.chat_history = []  # 存储对话历史
    self.max_history = max_history  # 历史长度限制
```

#### super() 的作用

```python
# super() 返回父类的代理对象
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)  # 调用父类 __init__
        self.age = age

child = Child("张三", 18)
print(child.name)  # "张三" (来自父类)
print(child.age)   # 18 (子类添加)
```

#### 历史数据结构

```python
# 使用列表存储元组
chat_history = [
    ("问题1", "答案1"),
    ("问题2", "答案2"),
    ("问题3", "答案3")
]

# 为什么用元组？
# 1. 不可变：保证历史不被意外修改
# 2. 高效：比列表更节省内存
# 3. 语义清晰：(问题, 答案) 配对明确
```

#### max_history 的作用

```python
# 问题：无限制的历史会导致什么？
chat_history = [...]  # 100 轮对话
# ❌ 占用大量内存
# ❌ Prompt 过长，超过 LLM token 限制
# ❌ 降低生成速度

# 解决：限制历史长度
max_history = 10  # 只保留最近 10 轮
# ✅ 内存可控
# ✅ Prompt 长度合理
# ✅ 保留最相关的上下文
```

**如何选择 max_history？**

| 场景 | 建议值 | 理由 |
|------|--------|------|
| 简单问答 | 3-5 | 对话简短，不需要太多历史 |
| 技术支持 | 5-10 | 需要较多上下文，但不会太长 |
| 复杂对话 | 10-20 | 需要丰富的历史信息 |
| 测试环境 | 100+ | 不限制，便于调试 |

### 5.4 默认模板（带历史）

```python
def _get_default_template(self) -> str:
    template = """你是一个智能助手。基于以下对话历史和上下文信息回答用户的问题。

    对话历史：
    {history}

    上下文信息：
    {context}

    当前问题：
    {question}

    请基于对话历史和上下文信息回答。如果问题与之前的对话相关，请考虑历史信息。

    你的回答："""
    return template
```

**与父类模板的对比**：

```python
# 父类模板
"""
上下文信息：{context}
用户问题：{question}
"""
# 占位符：2 个

# 子类模板
"""
对话历史：{history}  # ← 新增
上下文信息：{context}
当前问题：{question}
"""
# 占位符：3 个
```

**为什么历史放在最前面？**

```python
# 信息的时间顺序
对话历史 → 上下文信息 → 当前问题
(过去)    (相关知识)    (现在)

# LLM 理解逻辑
# 1. 先了解之前谈论了什么
# 2. 再看有哪些相关文档
# 3. 最后理解当前问题
```

### 5.5 query() 方法（带历史）

```python
def query(self, question: str, k: int = 4) -> str:
    try:
        # 步骤 1: 检索文档
        docs = self.vectorstore.similarity_search(question, k=k)
        if not docs:
            answer = "抱歉，我在知识库中没有找到相关信息。"
            self._add_to_history(question, answer)  # ← 保存到历史
            return answer
        
        # 步骤 2: 组装上下文
        context = "\n\n".join([
            f"[文档{i}]\n{doc.page_content}"
            for i, doc in enumerate(docs, 1)
        ])
        
        # 步骤 3: 格式化历史（关键！）
        history = self._format_history()
        
        # 步骤 4: 组装 Prompt（3 个占位符）
        prompt = self.prompt_template.format(
            history=history,    # ← 历史信息
            context=context,
            question=question
        )
        
        # 步骤 5: 生成答案
        answer = self.llm.generate(prompt)
        
        # 步骤 6: 保存到历史（关键！）
        self._add_to_history(question, answer)
        
        return answer
        
    except Exception as e:
        raise Exception(f"对话 RAG 查询失败: {str(e)}") from e
```

**关键区别**：

| 步骤 | RAGChain | ConversationalRAGChain |
|------|----------|----------------------|
| 检索 | ✅ 相同 | ✅ 相同 |
| 组装上下文 | ✅ 相同 | ✅ 相同 |
| **格式化历史** | ❌ 无 | ✅ **新增** |
| **Prompt 组装** | 2 个占位符 | **3 个占位符** |
| 生成 | ✅ 相同 | ✅ 相同 |
| **保存历史** | ❌ 无 | ✅ **新增** |

### 5.6 _format_history() 方法

```python
def _format_history(self) -> str:
    # 边界情况：没有历史
    if not self.chat_history:
        return "（这是第一轮对话）"
    
    # 格式化历史
    history_parts = []
    for i, (q, a) in enumerate(self.chat_history, 1):
        history_parts.append(f"用户: {q}")
        history_parts.append(f"助手: {a}")
    
    return "\n".join(history_parts)
```

**输入输出示例**：

```python
# 输入（内部数据结构）
chat_history = [
    ("Python 是什么？", "Python 是一种编程语言"),
    ("它有什么特点？", "Python 简洁易学")
]

# 输出（格式化字符串）
"""
用户: Python 是什么？
助手: Python 是一种编程语言
用户: 它有什么特点？
助手: Python 简洁易学
"""
```

**为什么这样设计？**

```python
# ✅ 清晰的对话结构
"用户: ..." 和 "助手: ..." 明确区分角色

# ✅ 时间顺序
最早的对话在前，最新的在后

# ✅ LLM 友好
LLM 能够理解这种格式，知道谁说了什么
```

### 5.7 _add_to_history() 方法

```python
def _add_to_history(self, question: str, answer: str):
    # 添加新对话
    self.chat_history.append((question, answer))
    
    # 保持历史长度限制
    if len(self.chat_history) > self.max_history:
        self.chat_history.pop(0)  # 删除最早的对话
```

**FIFO 策略（先进先出）**：

```python
# 假设 max_history = 3
chat_history = []

# 第 1 轮
_add_to_history("Q1", "A1")
# chat_history = [("Q1", "A1")]

# 第 2 轮
_add_to_history("Q2", "A2")
# chat_history = [("Q1", "A1"), ("Q2", "A2")]

# 第 3 轮
_add_to_history("Q3", "A3")
# chat_history = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]

# 第 4 轮（超过限制）
_add_to_history("Q4", "A4")
# chat_history.pop(0)  # 删除 ("Q1", "A1")
# chat_history = [("Q2", "A2"), ("Q3", "A3"), ("Q4", "A4")]
```

**为什么删除最早的？**

```python
# 假设场景：技术支持对话
chat_history = [
    ("打印机型号", "HP LaserJet"),     # 旧信息
    ("什么错误？", "卡纸"),            # 旧信息
    ("位置在哪？", "纸盒"),            # 旧信息
    ("如何解决？", "...")              # 最新
]

# 第 10 轮对话时
# 第 1 轮的信息（打印机型号）可能不再相关
# 最近的对话（如何解决）更重要
# → 删除最早的，保留最近的
```

### 5.8 stream_query() 方法（带历史）

```python
def stream_query(self, question: str, k: int = 4):
    try:
        # 步骤 1: 检索
        relevant_docs = self.vectorstore.similarity_search(question, k=k)
        if not relevant_docs:
            answer = "抱歉，我在知识库中没有找到相关信息。"
            self._add_to_history(question, answer)
            yield answer
            return
        
        # 步骤 2: 组装上下文
        context = "\n\n".join([...])
        
        # 步骤 3: 格式化历史
        history = self._format_history()
        
        # 步骤 4: 组装 Prompt
        prompt = self.prompt_template.format(
            history=history,
            context=context,
            question=question
        )
        
        # 步骤 5: 流式生成并收集答案
        answer_parts = []
        for chunk in self.llm.stream_generate(prompt):
            answer_parts.append(chunk)  # ← 收集片段
            yield chunk
        
        # 步骤 6: 保存完整答案到历史
        full_answer = "".join(answer_parts)  # ← 拼接完整答案
        self._add_to_history(question, full_answer)
        
    except Exception as e:
        raise Exception(f"流式对话 RAG 查询失败: {str(e)}") from e
```

**关键点：收集完整答案**

```python
# 为什么需要收集？
for chunk in stream_generate(...):
    print(chunk, end="")  # 打印片段
    # chunk 可能是："Python", " 是", "一种", "编程", "语言"

# 问题：如何保存完整答案到历史？
# 解决：收集所有片段

answer_parts = []
for chunk in stream_generate(...):
    answer_parts.append(chunk)
    yield chunk

full_answer = "".join(answer_parts)  # "Python 是一种编程语言"
_add_to_history(question, full_answer)
```

### 5.9 clear_history() 方法

```python
def clear_history(self):
    self.chat_history = []
```

**使用场景**：

```python
conv_rag = ConversationalRAGChain(llm, vectorstore)

# 对话 1：讨论 Python
conv_rag.query("Python 是什么？")
conv_rag.query("它有什么特点？")

# 主题切换
conv_rag.clear_history()  # 清空历史

# 对话 2：讨论 Java
conv_rag.query("Java 是什么？")
# 不会受之前 Python 对话的影响
```

---

## 六、技术细节与实现原理

### 6.1 向量检索原理

#### 语义相似度搜索

```python
# 步骤 1: 将问题转换为向量
question = "Python 是什么？"
question_vector = embedding_model.embed_query(question)
# question_vector = [0.1, 0.3, -0.2, ..., 0.5]  # 1536 维

# 步骤 2: 计算与文档向量的相似度
doc_vectors = vectorstore.get_all_vectors()
# doc_vectors = [
#     [0.1, 0.2, ...],  # 文档 1
#     [0.5, 0.6, ...],  # 文档 2
#     ...
# ]

# 步骤 3: 计算相似度（余弦相似度或 L2 距离）
similarities = [
    cosine_similarity(question_vector, doc_vec)
    for doc_vec in doc_vectors
]

# 步骤 4: 返回 Top-K
top_k_indices = argsort(similarities)[-k:]
relevant_docs = [documents[i] for i in top_k_indices]
```

#### 相似度度量

**余弦相似度**：

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 值域：[-1, 1]
# 1: 完全相同
# 0: 正交（无关）
# -1: 完全相反
```

**L2 距离（欧几里得距离）**：

```python
def l2_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# 值域：[0, ∞]
# 0: 完全相同
# 越大越不相似
```

### 6.2 Prompt 工程原理

#### 为什么 Prompt 重要？

```python
# ❌ 差的 Prompt
prompt = f"问题：{question}"
# 结果：LLM 可能编造答案

# ✅ 好的 Prompt
prompt = f"""
基于以下上下文回答，不要编造信息：
上下文：{context}
问题：{question}
"""
# 结果：LLM 基于上下文回答，更准确
```

#### Prompt 设计最佳实践

| 要素 | 作用 | 示例 |
|------|------|------|
| **角色定位** | 告诉 LLM 它是谁 | "你是专业助手" |
| **任务说明** | 明确要做什么 | "基于上下文回答" |
| **边界条件** | 处理特殊情况 | "如果没有信息，说不知道" |
| **输出格式** | 规范输出结构 | "使用要点列举" |
| **禁止事项** | 避免不当行为 | "不要编造信息" |

### 6.3 流式生成原理

#### 为什么能流式生成？

```python
# LLM 生成过程
# 1. Token-by-token 生成
"Python" → " 是" → "一种" → "编程" → "语言"

# 2. 每生成一个 token，立即返回
for token in llm.generate_tokens(prompt):
    yield token  # 不等待完整生成

# 3. 调用者实时获取
for chunk in stream_query(...):
    print(chunk, end="")  # 实时显示
```

#### 生成器的内存优势

```python
# ❌ 非流式：占用大量内存
def generate_all():
    result = ""
    for i in range(10000):
        result += generate_token(i)  # 拼接字符串
    return result  # 整个字符串在内存中

# ✅ 流式：内存友好
def generate_stream():
    for i in range(10000):
        yield generate_token(i)  # 只保存当前 token
```

### 6.4 异常处理设计

```python
try:
    # 可能出错的代码
    result = risky_operation()
except Exception as e:
    # 包装异常，提供更多上下文
    raise Exception(f"操作失败：{str(e)}") from e
    #                ↑                      ↑
    #           友好的错误信息          保留原始堆栈
```

**为什么这样设计？**

```python
# ❌ 直接抛出原始异常
def query(self, question):
    docs = self.vectorstore.similarity_search(question)
    # 如果出错，用户看到：
    # "ConnectionError: Failed to connect to FAISS"
    # 用户不知道是在 RAG 的哪个步骤出错

# ✅ 包装异常
def query(self, question):
    try:
        docs = self.vectorstore.similarity_search(question)
    except Exception as e:
        raise Exception(f"RAG 查询失败：{str(e)}") from e
    # 用户看到：
    # "Exception: RAG 查询失败：ConnectionError: Failed to connect to FAISS"
    # 清楚知道是 RAG 查询出错，原因是连接失败
```

---

## 七、设计模式与最佳实践

### 7.1 设计模式

#### 1. 策略模式（Strategy Pattern）

```python
# 不同的 LLM 策略
class RAGChain:
    def __init__(self, llm: BaseLLM, ...):
        self.llm = llm  # 依赖抽象接口

# 可以切换不同的策略
rag1 = RAGChain(llm=OpenAILLM(...))   # 策略 1
rag2 = RAGChain(llm=OllamaLLM(...))   # 策略 2
# 使用方式完全相同
```

#### 2. 模板方法模式（Template Method Pattern）

```python
# 父类定义算法骨架
class RAGChain:
    def query(self, question):
        docs = self._retrieve(question)      # 步骤 1
        context = self._format(docs)         # 步骤 2
        prompt = self._assemble(context, ...)  # 步骤 3
        return self._generate(prompt)        # 步骤 4

# 子类重写特定步骤
class ConversationalRAGChain(RAGChain):
    def _assemble(self, context, question):
        # 重写步骤 3，添加历史处理
        history = self._format_history()
        return self.prompt_template.format(
            history=history,
            context=context,
            question=question
        )
```

#### 3. 装饰器模式（Decorator Pattern）的思想

```python
# RAG 是对 LLM 的增强
# LLM: question → answer
# RAG: question → retrieve → augment → LLM → answer
#                    ↑         ↑
#              装饰器添加的功能
```

### 7.2 最佳实践

#### 1. 参数调优指南

**k 值选择**：

```python
# 场景 1：精确问答
k = 3  # 少而精，避免噪声

# 场景 2：综合性问题
k = 10  # 更全面的信息

# 场景 3：后续使用重排序
k = 20  # 先多召回，再精排
```

**max_history 选择**：

```python
# 场景 1：简单问答
max_history = 5

# 场景 2：复杂对话
max_history = 20

# 场景 3：长期会话
max_history = 100
```

#### 2. Prompt 优化

```python
# ❌ 模糊的指令
"回答问题：{question}"

# ✅ 清晰的指令
"""
仅基于以下上下文回答，不要编造信息：
上下文：{context}
问题：{question}
"""

# ✅✅ 更详细的指令
"""
你是专业助手。基于上下文回答：
上下文：{context}
问题：{question}

要求：
1. 仅基于上下文
2. 没有信息时明确说明
3. 使用要点列举
"""
```

#### 3. 错误处理

```python
# ✅ 边界检查
if not relevant_docs:
    return "抱歉，未找到相关信息"

# ✅ 异常包装
try:
    result = operation()
except Exception as e:
    raise Exception(f"操作失败：{str(e)}") from e

# ✅ 用户友好的错误信息
"RAG 查询失败：..."  # 而不是 "Exception in line 123"
```

---

## 八、常见问题与优化

### 8.1 常见问题

#### 问题 1：检索不到相关文档

**原因**：
- 向量库为空
- 查询与文档语义差距太大
- k 值设置不合理

**解决**：

```python
# 1. 检查向量库
if vectorstore.vectorstore is None:
    print("向量库为空，请先添加文档")

# 2. 使用带分数的搜索
docs_with_scores = vectorstore.similarity_search_with_score(
    question, k=10
)
for doc, score in docs_with_scores:
    print(f"分数：{score}，内容：{doc.page_content[:50]}...")

# 3. 降低检索阈值
k = 10  # 增加检索数量
```

#### 问题 2：答案质量不好

**原因**：
- Prompt 设计不合理
- 检索到的文档不相关
- k 值过大或过小

**解决**：

```python
# 1. 优化 Prompt
better_template = """
基于上下文，详细回答问题。不要编造信息。
上下文：{context}
问题：{question}
"""

# 2. 调整 k 值
# 从 k=4 改为 k=3 或 k=5

# 3. 添加重排序
# 先检索 20 个，再用 reranker 筛选 top 5
```

#### 问题 3：Token 超限

**原因**：
- k 太大，文档过多
- 单个文档太长
- 历史太长

**解决**：

```python
# 1. 减少 k
k = 3

# 2. 截断文档
contexts = [
    doc.page_content[:500]  # 只取前 500 字符
    for doc in relevant_docs
]

# 3. 限制历史
max_history = 5  # 减少历史轮数
```

#### 问题 4：流式输出不保存历史

**原因**：忘记收集完整答案

**解决**：

```python
# ✅ 正确做法
answer_parts = []
for chunk in self.llm.stream_generate(prompt):
    answer_parts.append(chunk)  # 收集
    yield chunk

full_answer = "".join(answer_parts)
self._add_to_history(question, full_answer)  # 保存完整答案
```

### 8.2 性能优化

#### 1. 批量处理

```python
# ❌ 逐个查询
answers = []
for question in questions:
    answer = rag_chain.query(question)
    answers.append(answer)

# ✅ 批量检索（如果支持）
all_docs = vectorstore.batch_search(questions)
# 然后批量生成
```

#### 2. 缓存优化

```python
from functools import lru_cache

class RAGChain:
    @lru_cache(maxsize=100)
    def _retrieve(self, question: str):
        """缓存检索结果"""
        return self.vectorstore.similarity_search(question)
```

#### 3. 异步处理

```python
import asyncio

class AsyncRAGChain(RAGChain):
    async def query_async(self, question: str):
        """异步查询"""
        # 异步检索
        docs = await asyncio.to_thread(
            self.vectorstore.similarity_search,
            question
        )
        # 异步生成
        answer = await asyncio.to_thread(
            self.llm.generate,
            prompt
        )
        return answer
```

---

## 九、完整使用示例

### 9.1 基本 RAG 使用

```python
from src.core.llm.base import OpenAILLM
from src.core.vectorstore.base import FAISSVectorStore
from src.core.chain.rag_chain import RAGChain
from langchain.schema import Document

# 步骤 1: 初始化组件
llm = OpenAILLM(model_name="gpt-3.5-turbo", temperature=0.7)
vectorstore = FAISSVectorStore(persist_directory="./data/vectors")

# 步骤 2: 添加文档到知识库
documents = [
    Document(page_content="Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"),
    Document(page_content="Python 以其简洁的语法和强大的标准库而闻名。"),
    Document(page_content="Python 支持多种编程范式，包括面向对象、函数式和过程式编程。")
]
vectorstore.add_documents(documents)

# 步骤 3: 创建 RAG Chain
rag_chain = RAGChain(llm=llm, vectorstore=vectorstore)

# 步骤 4: 查询
question = "Python 是什么时候创建的？"
answer = rag_chain.query(question)
print(f"问题：{question}")
print(f"答案：{answer}")
```

### 9.2 流式 RAG 使用

```python
# 流式查询
print("问题：详细介绍 Python 的特点")
print("答案：", end="", flush=True)

for chunk in rag_chain.stream_query("详细介绍 Python 的特点"):
    print(chunk, end="", flush=True)

print()  # 换行
```

### 9.3 对话式 RAG 使用

```python
from src.core.chain.rag_chain import ConversationalRAGChain

# 创建对话式 RAG Chain
conv_rag = ConversationalRAGChain(
    llm=llm,
    vectorstore=vectorstore,
    max_history=10
)

# 第一轮对话
q1 = "谁创建了 Python？"
a1 = conv_rag.query(q1)
print(f"Q1: {q1}")
print(f"A1: {a1}\n")

# 第二轮对话（利用历史）
q2 = "他在什么时候创建的？"  # "他" 指代 Guido
a2 = conv_rag.query(q2)
print(f"Q2: {q2}")
print(f"A2: {a2}\n")

# 第三轮对话（流式）
q3 = "它有什么特点？"  # "它" 指代 Python
print(f"Q3: {q3}")
print("A3: ", end="", flush=True)
for chunk in conv_rag.stream_query(q3):
    print(chunk, end="", flush=True)
print()

# 清空历史，开始新对话
conv_rag.clear_history()
```

### 9.4 自定义 Prompt 模板

```python
# 定义自定义模板
custom_template = """
你是一个教学助手，擅长用简单语言解释复杂概念。

参考资料：
{context}

学生问题：
{question}

请用通俗易懂的语言回答，适合初学者理解。

你的回答：
"""

# 使用自定义模板
rag_custom = RAGChain(
    llm=llm,
    vectorstore=vectorstore,
    prompt_template=custom_template
)

answer = rag_custom.query("什么是面向对象编程？")
print(answer)
```

### 9.5 完整的应用示例

```python
"""
完整的 RAG 应用：技术文档问答系统
"""

def build_knowledge_base():
    """构建知识库"""
    vectorstore = FAISSVectorStore("./data/tech_docs")
    
    # 加载文档
    docs = [
        Document(page_content="..."),
        # ... 更多文档
    ]
    
    vectorstore.add_documents(docs)
    vectorstore.persist()
    return vectorstore

def chat_loop():
    """对话循环"""
    # 初始化
    llm = OpenAILLM("gpt-3.5-turbo")
    vectorstore = FAISSVectorStore("./data/tech_docs")
    conv_rag = ConversationalRAGChain(llm, vectorstore)
    
    print("技术文档助手已启动。输入 'quit' 退出，'clear' 清空历史。")
    
    while True:
        question = input("\n你的问题：")
        
        if question.lower() == 'quit':
            break
        elif question.lower() == 'clear':
            conv_rag.clear_history()
            print("历史已清空")
            continue
        
        # 流式输出答案
        print("助手：", end="", flush=True)
        for chunk in conv_rag.stream_query(question):
            print(chunk, end="", flush=True)
        print()

if __name__ == "__main__":
    # 首次运行：构建知识库
    # build_knowledge_base()
    
    # 启动对话
    chat_loop()
```

---

## 十、总结

### 10.1 核心要点回顾

| 概念 | 核心内容 |
|------|---------|
| **RAG 本质** | 检索 + 生成，基于真实文档回答 |
| **RAGChain** | 基础类，实现核心 RAG 流程 |
| **ConversationalRAGChain** | 扩展类，添加对话历史管理 |
| **流式生成** | 使用生成器，逐个产出文本片段 |
| **Prompt 工程** | 企业级模板设计，确保答案质量 |

### 10.2 设计理念

1. **单一职责**：每个类/方法只做一件事
2. **依赖倒置**：依赖抽象接口，不依赖具体实现
3. **开闭原则**：对扩展开放，对修改封闭
4. **用户友好**：提供默认配置，支持自定义

### 10.3 学习建议

1. **理解原理**：先理解 RAG 的工作流程
2. **动手实践**：按照示例代码实际运行
3. **阅读源码**：查看每个方法的详细注释
4. **调试优化**：使用调试工具观察执行过程
5. **扩展创新**：尝试添加新功能（如重排序）

### 10.4 下一步

完成 RAG Chain 后，可以：
1. 实现文档加载器（PDF、Word、Markdown）
2. 实现知识库管理模块
3. 学习 LangGraph 实现 Agent
4. 构建完整的 RAG 应用（Web 界面）

---

## 附录

### A. 参考资源

- [LangChain 官方文档](https://python.langchain.com/)
- [RAG 论文](https://arxiv.org/abs/2005.11401)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### B. 常用命令

```bash
# 测试 RAG Chain
python examples/basic_rag_example.py

# 查看模块结构
tree src/core/chain/

# 运行测试
pytest tests/test_rag_chain.py
```

### C. 术语表

| 术语 | 含义 |
|------|------|
| RAG | Retrieval-Augmented Generation，检索增强生成 |
| LLM | Large Language Model，大语言模型 |
| Embedding | 将文本转换为向量的过程 |
| VectorStore | 向量数据库，存储和检索向量 |
| Prompt | 给 LLM 的输入指令 |
| Token | 文本的最小单位（词或字符） |
| Generator | Python 生成器，使用 yield 的函数 |
| FIFO | First In First Out，先进先出 |

---

**文档版本**：1.0  
**最后更新**：2025-01-08  
**作者**：HuahuaChat 项目团队
