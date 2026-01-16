# RAG Chain 实现指南

## 一、任务概览

### 1.1 当前状态

在 `src/core/chain/rag_chain.py` 文件中，已经定义了 `RAGChain` 类的**框架**，包括：

1. **类结构**：`RAGChain` 类已创建
2. **方法签名**：`__init__()`, `_get_default_template()`, `query()`, `stream_query()` 方法已定义
3. **文档注释**：每个方法都有详细的注释说明
4. **TODO 标记**：标记了需要你实现的部分

**你的任务**：填充这些方法的实现，让 RAG 系统真正运行起来。

### 1.2 需要实现的内容

```python
# 当前状态
class RAGChain:
    def __init__(self, llm, vectorstore, prompt_template=None):
        # 框架已有，需要完善
        pass
    
    def _get_default_template(self) -> str:
        # 需要实现：返回默认 Prompt 模板
        pass
    
    def query(self, question: str, k: int = 4) -> str:
        # 需要实现：核心 RAG 逻辑
        raise NotImplementedError("你需要实现这个方法")
    
    def stream_query(self, question: str, k: int = 4):
        # 需要实现：流式查询
        raise NotImplementedError("你需要实现这个方法")

# 还有一个继承类
class ConversationalRAGChain(RAGChain):
    # 需要实现：带对话历史的 RAG
    pass
```

---

## 二、RAG 核心原理（必读）

### 2.1 什么是 RAG？

**RAG = Retrieval-Augmented Generation（检索增强生成）**

```
传统 LLM 问答：
用户问题 → LLM → 答案
问题：可能不准确、过时、编造信息

RAG 问答：
用户问题 → 检索相关文档 → 组装 Prompt（问题+文档） → LLM → 准确答案
           ↑
      向量数据库
```

### 2.2 RAG 工作流程

```python
# RAG 的完整流程（伪代码）
def rag_process(question):
    # 步骤1: 检索 - 从知识库找相关文档
    docs = vectorstore.similarity_search(question, k=4)
    # 返回: [Document("Python是编程语言"), Document("Python简洁易学"), ...]
    
    # 步骤2: 提取内容 - 获取文档文本
    contexts = [doc.page_content for doc in docs]
    # 返回: ["Python是编程语言", "Python简洁易学", ...]
    
    # 步骤3: 组装上下文 - 拼接文档
    context = "\n\n".join(contexts)
    # 返回: "Python是编程语言\n\nPython简洁易学\n\n..."
    
    # 步骤4: 组装 Prompt - 填入模板
    prompt = f"""基于以下上下文回答问题：
    
    上下文：
    {context}
    
    问题：{question}
    
    回答："""
    
    # 步骤5: LLM 生成 - 调用 LLM
    answer = llm.generate(prompt)
    # 返回: "Python是一种高级编程语言，以简洁易学著称..."
    
    return answer
```

### 2.3 为什么 RAG 有效？

| 问题 | 传统 LLM | RAG |
|------|---------|-----|
| 知识过时 | ❌ 只能回答训练数据中的内容 | ✅ 可以更新知识库 |
| 专业知识 | ❌ 缺乏专业领域知识 | ✅ 导入专业文档即可 |
| 信息幻觉 | ❌ 容易编造不存在的信息 | ✅ 基于真实文档回答 |
| 可追溯性 | ❌ 无法提供信息来源 | ✅ 可以返回引用文档 |

---

## 三、方法实现详解

### 3.1 方法1: `_get_default_template()`

#### 任务说明

**作用**：返回默认的 Prompt 模板字符串

**要求**：
- 必须包含 `{context}` 占位符（用于填充检索到的文档）
- 必须包含 `{question}` 占位符（用于填充用户问题）
- 提供清晰的指令，告诉 LLM 如何回答

#### 实现步骤

**步骤1：理解占位符**

```python
# 占位符的作用
template = "上下文：{context}\n问题：{question}"

# 使用 format() 填充
filled = template.format(
    context="Python是编程语言",
    question="什么是Python？"
)
# 结果: "上下文：Python是编程语言\n问题：什么是Python？"
```

**步骤2：设计模板结构**

一个好的 Prompt 模板应该包含：

1. **角色定位**：告诉 LLM 它是谁
2. **任务说明**：告诉 LLM 要做什么
3. **上下文信息**：`{context}` 占位符
4. **用户问题**：`{question}` 占位符
5. **回答要求**：边界条件、格式要求等

**步骤3：编写代码**

```python
def _get_default_template(self) -> str:
    """
    获取默认 Prompt 模板
    
    Returns:
        包含 {context} 和 {question} 占位符的模板字符串
    """
    template = """你是一个智能助手，请基于以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：
{question}

回答要求：
1. 如果上下文中包含答案，请详细、准确地回答
2. 如果上下文中没有相关信息，请明确说"根据提供的信息，我无法回答这个问题"
3. 不要编造或推测上下文中不存在的信息
4. 回答要简洁明了

你的回答："""
    
    return template
```

#### Prompt 设计技巧

**✅ 好的 Prompt 设计**：

```python
# 清晰的角色
"你是一个专业的技术文档助手"

# 明确的指令
"请基于以下上下文回答问题，不要编造信息"

# 边界处理
"如果上下文中没有答案，请说'我不知道'"

# 清晰的占位符
"{context}"  # 上下文
"{question}" # 问题
```

**❌ 差的 Prompt 设计**：

```python
# 没有约束，容易编造
"问题：{question}\n回答："

# 没有上下文
"回答问题：{question}"

# 指令不明确
"{context}\n{question}"
```

#### 知识点

**Python 三引号字符串**：

```python
# 单行字符串
s1 = "这是一行"

# 多行字符串（使用三引号）
s2 = """这是
多行
字符串"""

# 保持格式的多行字符串
template = """第一行
第二行
第三行"""

print(template)
# 输出：
# 第一行
# 第二行
# 第三行
```

**字符串格式化**：

```python
# 方法1: format()
template = "名字：{name}，年龄：{age}"
result = template.format(name="张三", age=25)
# 结果: "名字：张三，年龄：25"

# 方法2: f-string (Python 3.6+)
name = "张三"
age = 25
result = f"名字：{name}，年龄：{age}"
# 结果: "名字：张三，年龄：25"

# RAG 中使用 format()
prompt = template.format(context="...", question="...")
```

---

### 3.2 方法2: `__init__()`

#### 任务说明

**作用**：初始化 RAG Chain，保存必要的组件引用

**当前代码**：

```python
def __init__(
    self,
    llm,  # TODO: 类型应该是 BaseLLM
    vectorstore,  # TODO: 类型应该是 BaseVectorStore
    prompt_template: Optional[str] = None
):
    self.llm = llm
    self.vectorstore = vectorstore
    self.prompt_template = prompt_template or self._get_default_template()
```

#### 需要做什么？

**当前代码已经基本完成，你需要**：

1. **添加类型提示**：明确参数类型
2. **理解逻辑**：理解 `or` 运算符的用法

#### 完善后的代码

```python
# 第1步：在文件开头导入类型
from src.core.llm.base import BaseLLM
from src.core.vectorstore.base import BaseVectorStore

# 第2步：添加类型提示
def __init__(
    self,
    llm: BaseLLM,                      # 添加类型提示
    vectorstore: BaseVectorStore,      # 添加类型提示
    prompt_template: Optional[str] = None
):
    """
    初始化 RAG Chain
    
    Args:
        llm: LLM 实例（OpenAILLM 或 OllamaLLM）
        vectorstore: 向量存储实例（FAISSVectorStore）
        prompt_template: 自定义 Prompt 模板（可选）
                       如果不提供，使用默认模板
    """
    # 保存 LLM 引用
    self.llm = llm
    
    # 保存 VectorStore 引用
    self.vectorstore = vectorstore
    
    # 设置 Prompt 模板
    # 如果用户提供了 prompt_template，使用用户的
    # 否则使用 _get_default_template() 返回的默认模板
    self.prompt_template = prompt_template or self._get_default_template()
```

#### 知识点

**Optional 类型提示**：

```python
from typing import Optional

# Optional[str] 表示：可以是 str，也可以是 None
def func(arg: Optional[str] = None):
    if arg is None:
        print("参数为空")
    else:
        print(f"参数值：{arg}")

# 等价于
def func(arg: str | None = None):
    pass
```

**逻辑 or 运算符**：

```python
# x or y：如果 x 为真，返回 x；否则返回 y
value = user_input or default_value

# 等价于：
if user_input:
    value = user_input
else:
    value = default_value

# 示例
name = "" or "匿名用户"        # 返回 "匿名用户"
name = "张三" or "匿名用户"    # 返回 "张三"
name = None or "匿名用户"      # 返回 "匿名用户"
```

---

### 3.3 方法3: `query()` —— 核心方法 ⭐

#### 任务说明

**作用**：执行完整的 RAG 查询流程

**这是整个项目最核心的方法！**

#### 实现步骤

**步骤1：检索相关文档**

```python
# 使用 vectorstore 的 similarity_search 方法
relevant_docs = self.vectorstore.similarity_search(
    query=question,  # 用户的问题
    k=k              # 检索 k 个最相关的文档
)

# relevant_docs 是一个列表，包含 Document 对象
# [Document(page_content="Python是编程语言", metadata={}), ...]
```

**步骤2：边界检查**

```python
# 如果没有检索到任何文档
if not relevant_docs:
    return "抱歉，我在知识库中没有找到相关信息。"
```

**步骤3：组装上下文**

```python
# 方法1：简单拼接
context = "\n\n".join([doc.page_content for doc in relevant_docs])

# 方法2：添加文档编号（推荐）
contexts = []
for i, doc in enumerate(relevant_docs, 1):
    context_piece = f"[文档{i}]\n{doc.page_content}"
    contexts.append(context_piece)

context = "\n\n".join(contexts)

# 结果示例：
# [文档1]
# Python是一种编程语言
#
# [文档2]
# Python简洁易学
```

**步骤4：组装 Prompt**

```python
# 使用 format() 填充模板中的占位符
prompt = self.prompt_template.format(
    context=context,      # 填充 {context}
    question=question     # 填充 {question}
)
```

**步骤5：调用 LLM 生成答案**

```python
# 调用 LLM 的 generate 方法
answer = self.llm.generate(prompt)
```

**步骤6：返回答案**

```python
return answer
```

#### 完整实现代码

```python
def query(self, question: str, k: int = 4) -> str:
    """
    执行 RAG 查询
    
    工作流程：
    1. 检索：从向量库中检索相关文档
    2. 组装：将文档内容组装成上下文
    3. 填充：将上下文和问题填入 Prompt 模板
    4. 生成：调用 LLM 生成答案
    5. 返回：返回生成的答案
    
    Args:
        question: 用户的问题
        k: 检索的文档数量（默认 4）
           建议范围：3-10
           - 太少：可能遗漏重要信息
           - 太多：引入噪声，增加 token 消耗
    
    Returns:
        生成的答案字符串
    
    Raises:
        Exception: 当查询失败时抛出异常
    """
    try:
        # ===== 步骤1: 检索相关文档 =====
        relevant_docs = self.vectorstore.similarity_search(
            query=question,
            k=k
        )
        
        # ===== 步骤2: 边界检查 =====
        if not relevant_docs:
            return "抱歉，我在知识库中没有找到相关信息。"
        
        # ===== 步骤3: 组装上下文 =====
        # 为每个文档添加编号
        contexts = []
        for i, doc in enumerate(relevant_docs, 1):
            # enumerate(list, 1) 从 1 开始编号
            context_piece = f"[文档{i}]\n{doc.page_content}"
            contexts.append(context_piece)
        
        # 用双换行连接所有文档
        context = "\n\n".join(contexts)
        
        # ===== 步骤4: 组装 Prompt =====
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # ===== 步骤5: LLM 生成答案 =====
        answer = self.llm.generate(prompt)
        
        # ===== 步骤6: 返回答案 =====
        return answer
        
    except Exception as e:
        # 捕获异常，提供友好的错误信息
        raise Exception(f"RAG 查询失败: {str(e)}") from e
```

#### 知识点详解

**enumerate() 函数**：

```python
# enumerate(iterable, start=0) 返回 (索引, 元素) 的元组
items = ["apple", "banana", "cherry"]

# 从 0 开始
for i, item in enumerate(items):
    print(f"{i}: {item}")
# 输出:
# 0: apple
# 1: banana
# 2: cherry

# 从 1 开始
for i, item in enumerate(items, 1):
    print(f"{i}: {item}")
# 输出:
# 1: apple
# 2: banana
# 3: cherry
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
```

**join() 方法**：

```python
# str.join(list) - 用指定字符串连接列表元素
items = ["a", "b", "c"]

"-".join(items)      # "a-b-c"
"\n".join(items)     # "a\nb\nc"
"\n\n".join(items)   # "a\n\nb\n\nc"

# RAG 中使用双换行，使文档之间有明显分隔
context = "\n\n".join(contexts)
```

**异常处理**：

```python
try:
    # 尝试执行的代码
    result = risky_operation()
except Exception as e:
    # 捕获异常
    # str(e) 获取错误信息
    # from e 保留原始异常的堆栈信息
    raise Exception(f"操作失败: {str(e)}") from e
```

---

### 3.4 方法4: `stream_query()` —— 流式查询

#### 任务说明

**作用**：流式执行 RAG 查询，逐个产出文本片段

**与 query() 的区别**：

| 特性 | query() | stream_query() |
|------|---------|----------------|
| 返回方式 | 一次性返回完整答案 | 逐个产出文本片段 |
| 用户体验 | 需要等待完整答案 | 实时看到生成过程 |
| 首字延迟 | 高 | 低 |
| 应用场景 | 短答案、批量处理 | 长答案、交互式对话 |

#### 实现步骤

**流式查询的特点**：

```python
# query() - 一次性返回
answer = rag_chain.query("什么是 Python？")
print(answer)  # 等待完整答案后输出

# stream_query() - 逐个产出
for chunk in rag_chain.stream_query("什么是 Python？"):
    print(chunk, end="", flush=True)  # 实时输出，打字机效果
```

**关键：使用生成器（Generator）**

```python
def stream_query(self, question: str, k: int = 4):
    # 步骤1-3：与 query() 相同（检索、组装）
    # ...
    
    # 步骤4：使用 yield 产出文本片段
    for chunk in self.llm.stream_generate(prompt):
        yield chunk  # yield 而不是 return
```

#### 完整实现代码

```python
def stream_query(self, question: str, k: int = 4):
    """
    流式执行 RAG 查询
    
    与 query() 的区别：
    - query(): 等待完整答案生成后返回
    - stream_query(): 逐个产出生成的文本片段
    
    优势：
    - 更好的用户体验（实时看到生成过程）
    - 更低的首字延迟
    - 可以实现打字机效果
    
    Args:
        question: 用户问题
        k: 检索文档数量
    
    Yields:
        str: 生成的文本片段
    
    Raises:
        Exception: 当查询失败时抛出异常
    """
    try:
        # ===== 步骤1: 检索文档 =====
        relevant_docs = self.vectorstore.similarity_search(
            query=question,
            k=k
        )
        
        # ===== 步骤2: 边界检查 =====
        if not relevant_docs:
            yield "抱歉，我在知识库中没有找到相关信息。"
            return  # 提前结束
        
        # ===== 步骤3: 组装上下文 =====
        contexts = [
            f"[文档{i}]\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs, 1)
        ]
        context = "\n\n".join(contexts)
        
        # ===== 步骤4: 组装 Prompt =====
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )
        
        # ===== 步骤5: 流式生成 =====
        # 使用 stream_generate() 而不是 generate()
        for chunk in self.llm.stream_generate(prompt):
            yield chunk  # 产出每个文本片段
        
    except Exception as e:
        raise Exception(f"流式 RAG 查询失败: {str(e)}") from e
```

#### 知识点详解

**生成器（Generator）**：

```python
# 普通函数 - 使用 return
def normal_function():
    return 1
    return 2  # 永远不会执行

result = normal_function()  # 1

# 生成器函数 - 使用 yield
def generator_function():
    yield 1
    yield 2
    yield 3

# 使用 for 循环遍历
for value in generator_function():
    print(value)
# 输出: 1, 2, 3
```

**yield vs return**：

```python
# return: 返回一次，函数结束
def func1():
    return "完整结果"

result = func1()
print(result)  # "完整结果"

# yield: 可以多次产出值
def func2():
    yield "片段1"
    yield "片段2"
    yield "片段3"

for chunk in func2():
    print(chunk)
# 输出:
# 片段1
# 片段2
# 片段3
```

**流式生成的实际应用**：

```python
# 场景1: 控制台打字机效果
print("问题:", question)
print("答案:", end=" ")

for chunk in rag_chain.stream_query(question):
    print(chunk, end="", flush=True)  # 不换行，实时输出
print()  # 最后换行

# 场景2: Web 应用 Server-Sent Events
def chat_api():
    def generate():
        for chunk in rag_chain.stream_query(question):
            yield f"data: {chunk}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')
```

**flush=True 的作用**：

```python
import time

# 没有 flush，输出会缓冲
for i in range(5):
    print(i, end="")
    time.sleep(1)
# 等待 5 秒后一次性输出: 01234

# 使用 flush=True，立即输出
for i in range(5):
    print(i, end="", flush=True)
    time.sleep(1)
# 每秒输出一个数字: 0...1...2...3...4
```

---

### 3.5 方法5: `ConversationalRAGChain` 类 —— 带对话历史

#### 任务说明

**作用**：实现带对话历史功能的 RAG Chain

**为什么需要对话历史？**

```python
# 没有对话历史
用户: "Python 是什么？"
助手: "Python 是一种编程语言。"

用户: "它有什么特点？"  # "它" 指什么？
助手: "它有什么特点？我不知道你说的是什么。" # ❌ 无法理解上下文

# 有对话历史
用户: "Python 是什么？"
助手: "Python 是一种编程语言。"

用户: "它有什么特点？"  # 系统知道 "它" 指 Python
助手: "Python 的特点包括：简洁易学、功能强大..." # ✅ 理解上下文
```

#### 实现步骤

**步骤1：继承 RAGChain**

```python
class ConversationalRAGChain(RAGChain):
    """带对话历史的 RAG Chain"""
    pass
```

**步骤2：添加历史存储**

```python
def __init__(self, llm, vectorstore, prompt_template=None, max_history=5):
    # 调用父类初始化
    super().__init__(llm, vectorstore, prompt_template)
    
    # 添加历史存储
    self.chat_history = []  # 存储 (问题, 答案) 元组
    self.max_history = max_history  # 最大保留历史轮数
```

**步骤3：修改 Prompt 模板**

```python
def _get_default_template(self) -> str:
    """带历史的 Prompt 模板"""
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

**步骤4：重写 query() 方法**

```python
def query(self, question: str, k: int = 4) -> str:
    # 检索文档
    docs = self.vectorstore.similarity_search(question, k=k)
    
    if not docs:
        answer = "抱歉，我在知识库中没有找到相关信息。"
        self._add_to_history(question, answer)
        return answer
    
    # 组装上下文
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 组装历史
    history = self._format_history()
    
    # 组装 Prompt（注意：现在有 history 参数）
    prompt = self.prompt_template.format(
        history=history,
        context=context,
        question=question
    )
    
    # 生成答案
    answer = self.llm.generate(prompt)
    
    # 添加到历史
    self._add_to_history(question, answer)
    
    return answer
```

**步骤5：实现辅助方法**

```python
def _format_history(self) -> str:
    """格式化历史"""
    if not self.chat_history:
        return "（这是第一轮对话）"
    
    history_parts = []
    for i, (q, a) in enumerate(self.chat_history, 1):
        history_parts.append(f"用户: {q}")
        history_parts.append(f"助手: {a}")
    
    return "\n".join(history_parts)

def _add_to_history(self, question: str, answer: str):
    """添加到历史"""
    self.chat_history.append((question, answer))
    
    # 限制历史长度
    if len(self.chat_history) > self.max_history:
        self.chat_history.pop(0)  # 移除最早的一条

def clear_history(self):
    """清空历史"""
    self.chat_history = []
```

#### 完整实现代码

```python
class ConversationalRAGChain(RAGChain):
    """
    带对话历史的 RAG Chain
    
    支持多轮对话，能够理解上下文引用
    
    示例：
        chain = ConversationalRAGChain(llm, vectorstore)
        
        # 第一轮
        answer1 = chain.query("什么是 Python？")
        
        # 第二轮（可以理解 "它" 指代 Python）
        answer2 = chain.query("它有什么特点？")
        
        # 清空历史
        chain.clear_history()
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        prompt_template: Optional[str] = None,
        max_history: int = 5
    ):
        """
        初始化对话式 RAG Chain
        
        Args:
            llm: LLM 实例
            vectorstore: 向量存储实例
            prompt_template: 自定义 Prompt 模板
            max_history: 最大保留的历史对话轮数
        """
        # 调用父类初始化
        super().__init__(llm, vectorstore, prompt_template)
        
        # 对话历史存储
        self.chat_history = []  # [(question, answer), ...]
        self.max_history = max_history
    
    def _get_default_template(self) -> str:
        """带历史的 Prompt 模板"""
        template = """你是一个智能助手。基于以下对话历史和上下文信息回答用户的问题。

对话历史：
{history}

上下文信息：
{context}

当前问题：
{question}

请基于对话历史和上下文信息回答。如果当前问题与之前的对话相关，请考虑历史信息。

你的回答："""
        return template
    
    def query(self, question: str, k: int = 4) -> str:
        """
        带历史的查询
        
        会自动管理对话历史
        """
        try:
            # 检索文档
            relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
            # 边界检查
            if not relevant_docs:
                answer = "抱歉，我在知识库中没有找到相关信息。"
                self._add_to_history(question, answer)
                return answer
            
            # 组装上下文
            context = "\n\n".join([
                f"[文档{i}]\n{doc.page_content}"
                for i, doc in enumerate(relevant_docs, 1)
            ])
            
            # 组装历史
            history = self._format_history()
            
            # 组装 Prompt
            prompt = self.prompt_template.format(
                history=history,
                context=context,
                question=question
            )
            
            # 生成答案
            answer = self.llm.generate(prompt)
            
            # 添加到历史
            self._add_to_history(question, answer)
            
            return answer
            
        except Exception as e:
            raise Exception(f"对话 RAG 查询失败: {str(e)}") from e
    
    def _format_history(self) -> str:
        """格式化对话历史"""
        if not self.chat_history:
            return "（这是第一轮对话）"
        
        history_parts = []
        for i, (q, a) in enumerate(self.chat_history, 1):
            history_parts.append(f"用户: {q}")
            history_parts.append(f"助手: {a}")
        
        return "\n".join(history_parts)
    
    def _add_to_history(self, question: str, answer: str):
        """添加到历史"""
        self.chat_history.append((question, answer))
        
        # 保持历史在限制内
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)  # 移除最早的一条
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
```

#### 知识点详解

**super() 函数**：

```python
class Parent:
    def __init__(self, name):
        self.name = name

class Child(Parent):
    def __init__(self, name, age):
        # 调用父类的 __init__
        super().__init__(name)
        self.age = age

# 使用
child = Child("张三", 18)
print(child.name)  # "张三" (来自父类)
print(child.age)   # 18 (子类添加)
```

**列表的 pop() 方法**：

```python
items = [1, 2, 3, 4, 5]

items.pop()     # 移除并返回最后一个: 5
items.pop(0)    # 移除并返回第一个: 1
items.pop(2)    # 移除并返回索引2的: 4

print(items)    # [2, 3]
```

**元组解包**：

```python
# 元组
pair = ("问题", "答案")

# 解包
question, answer = pair
print(question)  # "问题"
print(answer)    # "答案"

# 在循环中使用
history = [("Q1", "A1"), ("Q2", "A2")]
for q, a in history:
    print(f"Q: {q}, A: {a}")
```

---

## 四、完整代码汇总

### 4.1 完整的 rag_chain.py

```python
"""
RAG Chain 实现

这是 RAG 系统的核心组件
"""
from typing import List, Optional
from langchain.schema import Document

# 导入你实现的模块
from src.core.llm.base import BaseLLM
from src.core.vectorstore.base import BaseVectorStore


class RAGChain:
    """
    RAG (Retrieval-Augmented Generation) Chain
    
    核心功能：
    1. 从向量库检索相关文档
    2. 组装包含上下文的 Prompt
    3. 调用 LLM 生成答案
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        prompt_template: Optional[str] = None
    ):
        """
        初始化 RAG Chain
        
        Args:
            llm: LLM 实例
            vectorstore: 向量存储实例
            prompt_template: 自定义 Prompt 模板
        """
        self.llm = llm
        self.vectorstore = vectorstore
        self.prompt_template = prompt_template or self._get_default_template()
    
    def _get_default_template(self) -> str:
        """
        获取默认 Prompt 模板
        
        Returns:
            包含 {context} 和 {question} 占位符的模板字符串
        """
        template = """你是一个智能助手，请基于以下上下文信息回答用户的问题。

上下文信息：
{context}

用户问题：
{question}

回答要求：
1. 如果上下文中包含答案，请详细、准确地回答
2. 如果上下文中没有相关信息，请明确说"根据提供的信息，我无法回答这个问题"
3. 不要编造或推测上下文中不存在的信息
4. 回答要简洁明了

你的回答："""
        return template
    
    def query(self, question: str, k: int = 4) -> str:
        """
        执行 RAG 查询
        
        Args:
            question: 用户问题
            k: 检索的文档数量
            
        Returns:
            生成的答案
        """
        try:
            # 步骤1: 检索相关文档
            relevant_docs = self.vectorstore.similarity_search(
                query=question,
                k=k
            )
            
            # 步骤2: 边界检查
            if not relevant_docs:
                return "抱歉，我在知识库中没有找到相关信息。"
            
            # 步骤3: 组装上下文
            contexts = []
            for i, doc in enumerate(relevant_docs, 1):
                context_piece = f"[文档{i}]\n{doc.page_content}"
                contexts.append(context_piece)
            
            context = "\n\n".join(contexts)
            
            # 步骤4: 组装 Prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # 步骤5: LLM 生成答案
            answer = self.llm.generate(prompt)
            
            return answer
            
        except Exception as e:
            raise Exception(f"RAG 查询失败: {str(e)}") from e
    
    def stream_query(self, question: str, k: int = 4):
        """
        流式执行 RAG 查询
        
        Yields:
            生成的文本片段
        """
        try:
            # 步骤1: 检索文档
            relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
            # 步骤2: 边界检查
            if not relevant_docs:
                yield "抱歉，我在知识库中没有找到相关信息。"
                return
            
            # 步骤3: 组装上下文
            contexts = [
                f"[文档{i}]\n{doc.page_content}"
                for i, doc in enumerate(relevant_docs, 1)
            ]
            context = "\n\n".join(contexts)
            
            # 步骤4: 组装 Prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # 步骤5: 流式生成
            for chunk in self.llm.stream_generate(prompt):
                yield chunk
                
        except Exception as e:
            raise Exception(f"流式 RAG 查询失败: {str(e)}") from e


class ConversationalRAGChain(RAGChain):
    """
    带对话历史的 RAG Chain
    
    支持多轮对话，能够理解上下文引用
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        prompt_template: Optional[str] = None,
        max_history: int = 5
    ):
        """
        初始化对话式 RAG Chain
        
        Args:
            llm: LLM 实例
            vectorstore: 向量存储实例
            prompt_template: 自定义 Prompt 模板
            max_history: 最大保留的历史对话轮数
        """
        super().__init__(llm, vectorstore, prompt_template)
        self.chat_history = []
        self.max_history = max_history
    
    def _get_default_template(self) -> str:
        """带历史的 Prompt 模板"""
        template = """你是一个智能助手。基于以下对话历史和上下文信息回答用户的问题。

对话历史：
{history}

上下文信息：
{context}

当前问题：
{question}

请基于对话历史和上下文信息回答。如果当前问题与之前的对话相关，请考虑历史信息。

你的回答："""
        return template
    
    def query(self, question: str, k: int = 4) -> str:
        """带历史的查询"""
        try:
            # 检索文档
            relevant_docs = self.vectorstore.similarity_search(question, k=k)
            
            if not relevant_docs:
                answer = "抱歉，我在知识库中没有找到相关信息。"
                self._add_to_history(question, answer)
                return answer
            
            # 组装上下文
            context = "\n\n".join([
                f"[文档{i}]\n{doc.page_content}"
                for i, doc in enumerate(relevant_docs, 1)
            ])
            
            # 组装历史
            history = self._format_history()
            
            # 组装 Prompt
            prompt = self.prompt_template.format(
                history=history,
                context=context,
                question=question
            )
            
            # 生成答案
            answer = self.llm.generate(prompt)
            
            # 添加到历史
            self._add_to_history(question, answer)
            
            return answer
            
        except Exception as e:
            raise Exception(f"对话 RAG 查询失败: {str(e)}") from e
    
    def _format_history(self) -> str:
        """格式化对话历史"""
        if not self.chat_history:
            return "（这是第一轮对话）"
        
        history_parts = []
        for i, (q, a) in enumerate(self.chat_history, 1):
            history_parts.append(f"用户: {q}")
            history_parts.append(f"助手: {a}")
        
        return "\n".join(history_parts)
    
    def _add_to_history(self, question: str, answer: str):
        """添加到历史"""
        self.chat_history.append((question, answer))
        
        if len(self.chat_history) > self.max_history:
            self.chat_history.pop(0)
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
```

---

## 五、测试与验证

### 5.1 基本测试

```python
"""
测试 RAG Chain
"""
from src.core.llm.base import OpenAILLM
from src.core.vectorstore.base import FAISSVectorStore
from src.core.chain.rag_chain import RAGChain
from langchain.schema import Document


def test_basic_rag():
    """测试基本 RAG 功能"""
    # 初始化组件
    llm = OpenAILLM("gpt-3.5-turbo")
    vectorstore = FAISSVectorStore()
    
    # 添加测试文档
    docs = [
        Document(page_content="Python 是一种高级编程语言"),
        Document(page_content="Java 是面向对象的编程语言"),
        Document(page_content="LangChain 是用于构建 LLM 应用的框架")
    ]
    vectorstore.add_documents(docs)
    
    # 创建 RAG Chain
    rag_chain = RAGChain(llm=llm, vectorstore=vectorstore)
    
    # 测试查询
    question = "什么是 Python？"
    answer = rag_chain.query(question)
    
    print(f"问题: {question}")
    print(f"答案: {answer}")
    
    assert answer is not None
    assert len(answer) > 0


def test_stream_rag():
    """测试流式查询"""
    llm = OpenAILLM("gpt-3.5-turbo")
    vectorstore = FAISSVectorStore()
    
    docs = [Document(page_content="Python 简洁易学，功能强大")]
    vectorstore.add_documents(docs)
    
    rag_chain = RAGChain(llm=llm, vectorstore=vectorstore)
    
    print("问题: Python 的特点是什么？")
    print("答案: ", end="")
    
    for chunk in rag_chain.stream_query("Python 的特点是什么？"):
        print(chunk, end="", flush=True)
    print()


def test_conversational_rag():
    """测试对话式 RAG"""
    from src.core.chain.rag_chain import ConversationalRAGChain
    
    llm = OpenAILLM("gpt-3.5-turbo")
    vectorstore = FAISSVectorStore()
    
    docs = [Document(page_content="Python 由 Guido van Rossum 创建于 1991 年")]
    vectorstore.add_documents(docs)
    
    chain = ConversationalRAGChain(llm=llm, vectorstore=vectorstore)
    
    # 第一轮
    q1 = "谁创建了 Python？"
    a1 = chain.query(q1)
    print(f"Q1: {q1}")
    print(f"A1: {a1}\n")
    
    # 第二轮（引用上下文）
    q2 = "他在什么时候创建的？"  # "他" 指代 Guido
    a2 = chain.query(q2)
    print(f"Q2: {q2}")
    print(f"A2: {a2}\n")


if __name__ == "__main__":
    print("=" * 50)
    print("测试基本 RAG")
    print("=" * 50)
    test_basic_rag()
    
    print("\n" + "=" * 50)
    print("测试流式 RAG")
    print("=" * 50)
    test_stream_rag()
    
    print("\n" + "=" * 50)
    print("测试对话式 RAG")
    print("=" * 50)
    test_conversational_rag()
```

---

## 六、常见问题与调试

### 6.1 问题1: 导入错误

**错误**：
```python
ModuleNotFoundError: No module named 'src.core.llm'
```

**解决**：
```python
# 在文件开头正确导入
from src.core.llm.base import BaseLLM
from src.core.vectorstore.base import BaseVectorStore
```

### 6.2 问题2: 检索不到文档

**原因**：向量库为空或查询不匹配

**调试**：
```python
# 检查向量库
if vectorstore.vectorstore is None:
    print("向量库为空！")

# 查看检索结果
docs = vectorstore.similarity_search(question, k=10)
print(f"检索到 {len(docs)} 个文档")
for doc in docs:
    print(f"- {doc.page_content[:50]}...")
```

### 6.3 问题3: Prompt 格式错误

**错误**：
```python
KeyError: 'context'
```

**原因**：模板中缺少占位符或 format() 参数不匹配

**解决**：
```python
# 确保模板包含所有占位符
template = """
上下文: {context}
问题: {question}
回答:"""

# 确保 format() 提供所有参数
prompt = template.format(
    context=context,   # 必须提供
    question=question  # 必须提供
)
```

---

## 七、总结

### 7.1 实现清单

- [ ] 修改文件开头的导入语句
- [ ] 实现 `_get_default_template()` 方法
- [ ] 完善 `__init__()` 方法的类型提示
- [ ] 实现 `query()` 方法（核心）
- [ ] 实现 `stream_query()` 方法
- [ ] 实现 `ConversationalRAGChain` 类
- [ ] 编写测试代码
- [ ] 测试完整流程

### 7.2 核心知识点

1. **RAG 原理**：检索 + 生成
2. **Prompt 设计**：清晰的指令和占位符
3. **流式生成**：yield 关键字和生成器
4. **对话历史**：继承和历史管理
5. **Python 语法**：enumerate, join, format, super

### 7.3 下一步

完成 RAG Chain 后，你可以：
1. 实现文档加载和切分功能
2. 实现知识库管理模块
3. 学习 LangGraph 实现 Agent
4. 构建完整的 RAG 应用

加油！这是项目的核心部分，实现完成后你就拥有了一个可用的 RAG 系统！