# Embedding 模块实现指南

## 一、Embedding 模块概述

### 1.1 什么是 Embedding？

**Embedding（嵌入）**是将文本转换为数值向量的过程。这些向量能够捕捉文本的语义信息，使得语义相似的文本在向量空间中距离更近。

**简单理解**：
- 文本："什么是 Python？" → 向量：[0.1, 0.3, -0.2, ..., 0.5]（1536 维）
- 文本："Python 是什么？" → 向量：[0.12, 0.28, -0.19, ..., 0.48]（相似）
- 文本："今天天气真好" → 向量：[0.8, -0.1, 0.3, ..., -0.2]（不相似）

### 1.2 Embedding 在 RAG 中的作用

**RAG 流程中的位置**：

```
文档 → 文本切分 → Embedding 向量化 → 存储到向量库 → 检索相似文档
                                                      ↑
用户问题 → Embedding 向量化 → 向量相似度搜索 ──────┘
```

**关键作用**：
1. **文档向量化**：将文档转换为向量，存储到向量库
2. **查询向量化**：将用户问题转换为向量，用于检索
3. **相似度计算**：通过向量距离找到最相关的文档

### 1.3 为什么需要两个方法？

**embed_query()** 和 **embed_documents()** 的区别：

| 特性 | embed_query() | embed_documents() |
|------|--------------|-------------------|
| 输入 | 单个字符串 | 字符串列表 |
| 输出 | 单个向量 `List[float]` | 向量列表 `List[List[float]]` |
| 使用场景 | 用户查询 | 文档入库 |
| 优化 | 针对查询优化 | 针对文档优化 |

**为什么分开？**
- 某些模型对查询和文档有不同的处理方式
- 查询通常较短，文档可能较长
- 可以分别优化性能

**实际使用**：
```python
# 文档入库时
documents = ["文档1", "文档2", "文档3"]
vectors = embedding.embed_documents(documents)  # 批量处理

# 用户查询时
query = "用户的问题"
query_vector = embedding.embed_query(query)  # 单个查询
```

---

## 二、架构设计

### 2.1 与 LLM 模块的对比

**相似点**：
- 都使用抽象接口（BaseEmbedding vs BaseLLM）
- 都支持多种提供商（OpenAI、本地模型等）
- 都封装底层实现细节

**不同点**：

| 特性 | LLM 模块 | Embedding 模块 |
|------|---------|---------------|
| 输入 | 字符串（prompt） | 字符串或字符串列表 |
| 输出 | 字符串（回答） | 向量（数值列表） |
| 调用方式 | invoke() / stream() | embed_query() / embed_documents() |
| 消息格式 | 需要 HumanMessage | 直接使用字符串 |
| 应用场景 | 文本生成 | 向量化 |

### 2.2 接口设计

```python
class BaseEmbedding(ABC):
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """单个查询向量化"""
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        pass
```

**设计优势**：
- 接口统一，可以切换不同的 Embedding 模型
- 支持批量处理（embed_documents）
- 区分查询和文档的处理

---

## 三、实现步骤详解

### 3.1 查找 LangChain 文档

**文档位置**：
- 主文档：https://python.langchain.com/docs/integrations/text_embedding/openai
- 搜索关键词：`OpenAIEmbeddings`, `embed_query`, `embed_documents`

**关键信息**：
1. 如何初始化 `OpenAIEmbeddings`
2. 如何使用 `embed_query()` 方法
3. 如何使用 `embed_documents()` 方法
4. 参数配置（model、api_key 等）

### 3.2 实现 OpenAIEmbedding 类

#### 步骤1：导入必要的模块

```python
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from typing import List
```

#### 步骤2：初始化方法

```python
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        # 获取 API Key
        api_key = os.getenv("OPENAI_API_KEY")
        
        # 初始化 OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings(
            model=model_name,  # 例如 "text-embedding-ada-002"
            openai_api_key=api_key
        )
```

**知识点**：
- `OpenAIEmbeddings` 是 LangChain 提供的类
- 不需要消息格式（不像 ChatOpenAI）
- 直接使用字符串即可

#### 步骤3：实现 embed_query()

```python
def embed_query(self, text: str) -> List[float]:
    """
    对单个查询文本进行向量化
    
    Args:
        text: 查询文本（字符串）
        
    Returns:
        向量表示（浮点数列表）
    """
    try:
        # 直接调用 OpenAIEmbeddings 的 embed_query 方法
        vector = self.embeddings.embed_query(text)
        return vector
    except Exception as e:
        raise Exception(f"向量化查询失败: {str(e)}") from e
```

**关键点**：
- `embed_query()` 接受字符串，返回 `List[float]`
- 不需要转换格式（不像 LLM 需要 HumanMessage）
- 直接调用底层方法即可

#### 步骤4：实现 embed_documents()

```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    """
    对文档列表进行向量化
    
    Args:
        texts: 文档文本列表
        
    Returns:
        向量列表（每个文档对应一个向量）
    """
    try:
        # 直接调用 OpenAIEmbeddings 的 embed_documents 方法
        vectors = self.embeddings.embed_documents(texts)
        return vectors
    except Exception as e:
        raise Exception(f"向量化文档失败: {str(e)}") from e
```

**关键点**：
- `embed_documents()` 接受字符串列表，返回向量列表
- 批量处理，效率更高
- 返回类型是 `List[List[float]]`（列表的列表）

### 3.3 完整实现示例

```python
"""
Embedding 基础接口定义
"""
from abc import ABC, abstractmethod
from typing import List
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class BaseEmbedding(ABC):
    """Embedding 基础接口"""
    
    def __init__(self, model_name: str):
        """
        初始化 Embedding 模型
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询文本进行向量化
        
        Args:
            text: 查询文本
            
        Returns:
            向量表示
        """
        pass
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行向量化
        
        Args:
            texts: 文档文本列表
            
        Returns:
            向量列表
        """
        pass


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding 实现"""
    
    def __init__(self, model_name: str):
        """
        初始化 OpenAI Embedding
        
        Args:
            model_name: 模型名称，例如 "text-embedding-ada-002"
        """
        super().__init__(model_name)
        
        # 获取 API Key
        api_key = os.getenv("OPENAI_API_KEY")
        
        # 初始化 OpenAIEmbeddings
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
    
    def embed_query(self, text: str) -> List[float]:
        """
        对单个查询文本进行向量化
        
        Args:
            text: 查询文本
            
        Returns:
            向量表示（浮点数列表）
        """
        try:
            # 直接调用 OpenAIEmbeddings 的 embed_query 方法
            vector = self.embeddings.embed_query(text)
            return vector
        except Exception as e:
            raise Exception(f"向量化查询失败: {str(e)}") from e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        对文档列表进行向量化
        
        Args:
            texts: 文档文本列表
            
        Returns:
            向量列表（每个文档对应一个向量）
        """
        try:
            # 直接调用 OpenAIEmbeddings 的 embed_documents 方法
            vectors = self.embeddings.embed_documents(texts)
            return vectors
        except Exception as e:
            raise Exception(f"向量化文档失败: {str(e)}") from e
```

---

## 四、关键知识点

### 4.1 向量是什么？

**向量（Vector）**是一组数值，表示文本在高维空间中的位置。

**示例**：
```python
# 文本："什么是 Python？"
vector = [0.1, 0.3, -0.2, 0.5, ..., 0.2]  # 1536 个数字（OpenAI ada-002）

# 向量的维度
print(len(vector))  # 1536

# 向量的类型
print(type(vector[0]))  # <class 'float'>
```

**为什么是浮点数？**
- 向量中的每个数字都是浮点数（float）
- 表示文本在不同维度上的"特征值"
- 通过计算向量距离来判断文本相似度

### 4.2 embed_query() vs embed_documents()

**为什么有两个方法？**

1. **语义区分**：
   - 查询通常是问题，较短
   - 文档通常是内容，较长
   - 某些模型对两者有不同的优化

2. **性能优化**：
   - `embed_documents()` 可以批量处理，效率更高
   - `embed_query()` 针对单个查询优化

3. **实际使用**：
   ```python
   # 文档入库（批量）
   documents = ["文档1", "文档2", "文档3"]
   doc_vectors = embedding.embed_documents(documents)
   # 返回: [[0.1, 0.2, ...], [0.3, 0.1, ...], [0.2, 0.4, ...]]
   
   # 用户查询（单个）
   query = "用户的问题"
   query_vector = embedding.embed_query(query)
   # 返回: [0.15, 0.25, ...]
   ```

### 4.3 返回类型说明

**embed_query()**：
```python
def embed_query(self, text: str) -> List[float]:
    # 输入: "什么是 Python？"
    # 输出: [0.1, 0.3, -0.2, ..., 0.5]  # 一个向量（列表）
    return vector  # List[float]
```

**embed_documents()**：
```python
def embed_documents(self, texts: List[str]) -> List[List[float]]:
    # 输入: ["文档1", "文档2", "文档3"]
    # 输出: [
    #   [0.1, 0.2, ...],  # 文档1的向量
    #   [0.3, 0.1, ...],  # 文档2的向量
    #   [0.2, 0.4, ...]   # 文档3的向量
    # ]
    return vectors  # List[List[float]] - 列表的列表
```

**理解嵌套列表**：
- `List[float]`：一个向量（一维列表）
- `List[List[float]]`：多个向量（二维列表）
- 外层列表：文档数量
- 内层列表：每个文档的向量

### 4.4 与 LLM 模块的区别

**重要区别**：

1. **不需要消息格式**：
   ```python
   # LLM 需要
   message = HumanMessage(content=prompt)
   messages = [message]
   response = llm.invoke(messages)
   
   # Embedding 不需要
   vector = embeddings.embed_query(text)  # 直接使用字符串
   ```

2. **返回类型不同**：
   ```python
   # LLM 返回字符串
   answer = llm.generate("问题")  # str
   
   # Embedding 返回向量
   vector = embedding.embed_query("问题")  # List[float]
   ```

3. **调用方式不同**：
   ```python
   # LLM 使用 invoke() 或 stream()
   response = llm.invoke(messages)
   
   # Embedding 使用 embed_query() 或 embed_documents()
   vector = embeddings.embed_query(text)
   ```

---

## 五、测试代码

### 5.1 基本测试

```python
# test_embedding.py
from src.core.embedding.base import OpenAIEmbedding

# 创建实例
embedding = OpenAIEmbedding(model_name="text-embedding-ada-002")

# 测试 embed_query
query = "什么是 Python？"
query_vector = embedding.embed_query(query)
print(f"查询向量维度: {len(query_vector)}")
print(f"向量前5个值: {query_vector[:5]}")

# 测试 embed_documents
documents = [
    "Python 是一种编程语言",
    "Java 是一种编程语言",
    "今天天气真好"
]
doc_vectors = embedding.embed_documents(documents)
print(f"文档数量: {len(doc_vectors)}")
print(f"每个向量维度: {len(doc_vectors[0])}")
```

### 5.2 相似度测试

```python
import numpy as np

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 测试相似度
query = "什么是 Python？"
doc1 = "Python 是一种编程语言"
doc2 = "今天天气真好"

query_vec = embedding.embed_query(query)
doc1_vec = embedding.embed_query(doc1)  # 注意：也可以用 embed_query
doc2_vec = embedding.embed_query(doc2)

# 计算相似度
similarity1 = cosine_similarity(query_vec, doc1_vec)
similarity2 = cosine_similarity(query_vec, doc2_vec)

print(f"查询与文档1的相似度: {similarity1:.4f}")  # 应该较高
print(f"查询与文档2的相似度: {similarity2:.4f}")  # 应该较低
```

---

## 六、常见问题

### Q1: embed_query() 和 embed_documents() 可以互换吗？

**A**: 技术上可以，但不推荐。
- `embed_query()` 可以处理单个字符串
- `embed_documents()` 可以处理列表（即使只有一个元素）
- 但语义上不同，应该按用途使用

### Q2: 向量维度是多少？

**A**: 取决于模型：
- OpenAI `text-embedding-ada-002`：1536 维
- OpenAI `text-embedding-3-small`：1536 维
- OpenAI `text-embedding-3-large`：3072 维

### Q3: 为什么向量是浮点数？

**A**: 
- 浮点数可以表示小数值
- 向量计算需要精确的数值
- 相似度计算（如余弦相似度）需要浮点数

### Q4: 可以本地运行 Embedding 吗？

**A**: 可以，使用本地模型：
- `sentence-transformers` 库
- `HuggingFaceEmbeddings`
- 不需要 API Key

---

## 七、实现检查清单

- [ ] 导入必要的模块（OpenAIEmbeddings, os, dotenv）
- [ ] 实现 `__init__()` 方法
  - [ ] 调用 `super().__init__(model_name)`
  - [ ] 获取 API Key
  - [ ] 初始化 `OpenAIEmbeddings`
- [ ] 实现 `embed_query()` 方法
  - [ ] 调用 `self.embeddings.embed_query(text)`
  - [ ] 返回 `List[float]`
  - [ ] 添加错误处理
- [ ] 实现 `embed_documents()` 方法
  - [ ] 调用 `self.embeddings.embed_documents(texts)`
  - [ ] 返回 `List[List[float]]`
  - [ ] 添加错误处理
- [ ] 编写测试代码
- [ ] 验证向量维度正确
- [ ] 测试相似度计算

---

## 八、下一步

完成 Embedding 模块后，你将：
1. ✅ 理解向量化的概念
2. ✅ 掌握 Embedding 的使用方法
3. ✅ 为向量存储模块做准备

**下一步**：实现向量存储模块（VectorStore），使用 Embedding 向量进行相似度检索。

---

## 九、参考资源

- [LangChain OpenAI Embeddings 文档](https://python.langchain.com/docs/integrations/text_embedding/openai)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [向量相似度计算](https://en.wikipedia.org/wiki/Cosine_similarity)

