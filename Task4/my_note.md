# Happy-LLM 学习笔记 Task 04: 预训练语言模型

> **课程来源**: [Datawhale Happy-LLM 官网](https://datawhalechina.github.io/happy-llm/)

> **我的笔记仓库**: [https://github.com/GuYan66/happy-llm.git](https://github.com/GuYan66/happy-llm.git)

---

在Transformer模型诞生后，NLP领域围绕其核心组件 (Encoder和Decoder) 形成了三种主流的预训练模型架构：

- Encoder-Only: 以 BERT 为代表，专注于自然语言理解（NLU）任务。
- Decoder-Only: 以 GPT 为代表，专注于自然语言生成（NLG）任务，是当前主流LLM的基础。
- Encoder-Decoder: 以 T5 为代表，继承了完整的Transformer结构，试图统一所有NLP任务。



##  1. Encoder-only PLM

### 1.1 BERT

BERT (Bidirectional Encoder Representations from Transformers) 是Google在2018年推出的里程碑式模型，它确立了“预训练+微调”范式在NLP领域的统治地位。

**(1) 核心思想**
Transformer 架构: BERT的基础是Transformer的Encoder部分，通过堆叠Encoder层来构建深层网络。

预训练+微调范式: 继承自ELMo，但在Transformer架构上引入更适合文本理解、能捕捉深层双向语义关系的预训练任务 MLM 将其效果发挥到极致。



**(2) 模型架构**

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/1-0.png" alt="图片描述" width="100%"/>
  <p>图1 BERT 模型结构</p>
</div>


**结构**: 

完全由Transformer的Encoder Layer堆叠而成。顶部附加一个任务相关的`prediction_heads`（通常是线性层+激活函数）用于微调。

**版本**:

- `BERT-base`: 12层Encoder, 768维隐藏层, 12个注意力头, 110M参数。
- `BERT-large`: 24层Encoder, 1024维隐藏层, 16个注意力头, 340M参数。

**输入表示**: 每个输入的token表示由三部分**相加**而成：

1. **Token Embeddings**: 词本身的嵌入表示。
2. **Segment Embeddings**: 用于区分句子对中的句子A和句子B（在NSP任务和某些下游任务中使用）。
3. **Position Embeddings**: 可学习的位置向量，用于表示词在序列中的位置。

**关键组件**:

- **Tokenizer**: 采用 **WordPiece** 作为子词切分算法，有效缓解OOV（未登录词）问题。
- **激活函数**: 首次推广使用**GELU (高斯误差线性单元)**，相比ReLU在负值区有更平滑的导数，性能更优。
- **位置编码**: 使用**可学习的绝对位置编码**，而非固定的`sin/cos`函数。



**(3) 预训练任务**

BERT更大的创新点在于其提出的MLM和NSP两个新的预训练任务上。预训练-微调范式的核心优势在于，通过将预训练和微调分离，完成一次预训练的模型可以仅通过微调应用在几乎所有下游任务上。

**传统预训练方法LM：**

预训练-微调的核心是需要极大的数据规模，因此预训练数据是从无监督的语料中获取。这也是为什么传统的预训练任务都是 LM 的原因——LM 使用上文预测下文的方式可以直接应用到任何文本中，对于任意文本，我们只需要将下文遮蔽将上文输入模型要求其预测就可以实现 LM 训练，因此互联网上所有文本语料都可以被用于预训练。

LM的缺陷：其直接拟合从左到右的语义关系，但忽略了双向的语义关系。基于此提出了MLM。


**MLM (Masked Language Model，掩码语言模型)** 

MLM 的思路是在一个文本序列中随机遮蔽部分 token，然后将所有未被遮蔽的 token 输入模型，要求模型根据输入预测被遮蔽的 token。例如，输入和输出可以是：

    输入：I <MASK> you because you are <MASK>
    输出：<MASK> - love; <MASK> - wonderful

**优势:**

- 模型可以利用被遮蔽的 token 的上文和下文一起理解语义来预测被遮蔽的 token，因此模型可以拟合双向语义，c从而更好地实现文本的理解。
- 无需对文本进行任何人为的标注，只需要对文本进行随机遮蔽即可，因此也可以利用互联网所有文本语料实现预训练。例如，BERT 的预训练就使用了足足 3300M 单词的语料。

**机制**：

在输入序列中随机选择15%的token进行用于遮蔽，然后让模型预测这些位置的原始token。但是这15%的token采用了下面的80-10-10策略。

- **80-10-10 策略**: 这是为了缓解预训练（输入包含`[MASK]`）和微调（输入不含`[MASK]`）之间的不一致性而设计的精妙策略。
- - **80%** 的概率被遮蔽，即被 `[MASK]`标记。
  - **10%** 的概率被替换为一个**随机**的token。这迫使模型不能只依赖于`[MASK]`标记本身，而必须学习上下文的表示。
  - **10%** 的概率**保持不变**。这使得模型需要学习对每一个输入token都进行表征，因为任何一个词都可能是被“篡改”的。



**NSP（Next Sentence Prediction，下一句预测）**

NSP 任务的核心目标是让模型判断一对句子是否是连续的上下文。这对于句级的自然语言理解（NLU）任务非常重要，尤其是问答匹配、自然语言推理等任务。

**核心思想：**NSP 强调的是**句子级的关系理解**，即判断两个句子是否连续。与传统的词级模型（如 MLM）不同，NSP 让模型学习句子之间的语义联系，从而帮助其在句级任务中取得更好的表现。

例如，输入和输入可以是：

    输入：
        Sentence A：I love you.
        Sentence B: Because you are wonderful.
    输出：
        1（是连续上下文）
    
    输入：
        Sentence A：I love you.
        Sentence B: Because today's dinner is so nice.
    输出：
        0（不是连续上下文）

其中，NSP 的正样本可以从无监督语料中随机抽取任意连续的句子，而负样本可以对句子打乱后随机抽取（只需要保证不要抽取到原本就连续的句子就行），因此也可以具有几乎无限量的训练数据。



### 1.2 RoBERTa

RoBERTa的核心贡献在于证明了BERT实际上是**训练不足（undertrained）**的，并通过一系列精细的优化显著提升了其性能。

**优化**:

1. **移除NSP任务**: 实验证明，移除NSP任务，仅使用MLM，并在一些场景下让输入跨文档（一个输入包括多个文档），性能反而更好。
2. **动态遮蔽 (Dynamic Masking)**: BERT在数据预处理时进行一次性静态遮蔽，而RoBERTa在模型训练的每个Epoch都为同一序列动态生成新的遮蔽模式，增加了训练数据的多样性和难度。
3. **更大规模的训练**:
   - **数据**: 使用了更大量的无监督语料进行预训练，共计160GB的数据（BERT为13GB）。
   - **Batch Size**: 使用了巨大的批次大小（8K），这被证明可以提升优化速度和最终性能。
   - **训练步数**: 训练了更长的时间（500K步 vs BERT的1M步但batch size小得多）。
4. **更大的BPE词表**: 使用了BPE（Byte Pair Encoding，字节对编码，是指以子词对作为分词的单位）作为Tokenizer 的编码策略。它用了50K的字节级BPE词表，而非BERT的30K WordPiece词表。



### 1.3 ALBERT

ALBERT的核心目标是在不牺牲（甚至提升）性能的前提下，**大幅度减少模型参数量**。

虽然 ALBERT 所提出的一些改进思想并没有在后续研究中被广泛采用，但其降低模型参数的方法及提出的新预训练任务 SOP 仍然对 NLP 领域提供了重要的参考意义。

**优化**:

1. **词嵌入参数分解 (Factorized Embedding Parameterization)**: 传统的Embedding层参数为 $V*H$（词表大小 * 隐藏层维度）。ALBERT将其分解为两个小矩阵 $V*E$ 和 $E*H$，其中 $E*H$，这使得词嵌入维度(E)与巨大的隐藏层维度(H)解耦，参数从 $V*H$ 降低到了 $V*E + E*H$。
2. **跨层参数共享 (Cross-layer Parameter Sharing)**: ALBERT让所有Transformer的Encoder Layer**共享同一套参数**（可以选择只共享注意力部分或FFN部分，或全部共享）。这极大地减少了模型参数，但计算量（前向传播的FLOPs）并未减少。
3. **SOP预训练任务 (Sentence-Order Prediction)**: 提出SOP作为NSP的替代品。SOP任务的负样本不再是随机句子，而是将同一文档中的两个连续句子的顺序颠倒。这比NSP更难，因为它迫使模型不仅要拟合两个句子之间的关系，更要学习其顺序关系。



## 2. Encoder-Decoder PLM

#### 2.1 T5

T5（Text-To-Text Transfer Transformer）是由 Google 提出的一种预训练语言模型，通过将所有 NLP 任务统一表示为文本到文本的转换问题，大大简化了模型设计和任务处理。

**(1) 核心思想**

提出并验证了**一个统一的框架可以解决所有NLP问题**。

**大一统思想**: 将**所有NLP任务都统一为文本到文本（Text-to-Text）的格式**。模型接收文本输入，并生成文本输出。

- **实现方式**: 通过在输入前添加**任务前缀**来指示模型执行何种任务。



**(2) 模型架构**

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/2-1.png" alt="图片描述" width="100%"/>
  <p>图2 T5 模型详细结构</p>
</div>

**结构**: 采用标准的**Encoder-Decoder**架构。

**创新点**:

- **归一化**: 采用了**RMSNorm (Root Mean Square Layer Normalization)**，通过计算每个神经元的均方根（Root Mean Square）来归一化每个隐藏层的激活值。它比LayerNorm更简洁高效，只有一个可学习的缩放参数 gamma，公式如下：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

其中：

- $x_i$ 是输入向量的第 $i$ 个元素
- $\gamma$ 是可学习的缩放参数
- $n$ 是输入向量的维度数量
- $\epsilon$ 是一个小常数，用于数值稳定性（以避免除以零的情况）

这种归一化有助于通过确保权重的规模不会变得过大或过小来稳定学习过程，这在具有许多层的深度学习模型中特别有用。

- **位置编码**: 采用了一种简化的相对位置编码，直接将位置偏差作为一个可学习的标量添加到注意力logit上。



**(3) 预训练任务**

采用了一种更具挑战性的**掩码填充（Span Corruption）**任务。

- **机制**: 不是像BERT那样mask单个token，而是随机mask掉连续的文本片段（span），然后让模型在输出端自回归地生成这些被mask掉的片段，并用特殊的哨兵token来区分不同的片段。



## 3. Decoder-Only PLM

Decoder-Only 就是目前大火的 LLM 的基础架构，目前所有的 LLM 基本都是 Decoder-Only 模型（RWKV、Mamba 等非 Transformer 架构除外）。



### 3.1 GPT系列

GPT (Generative Pre-Training Language Model) 是OpenAI团队于2018年发布的预训练语言模型，其发展历程表明了模型性能随着模型大小、数据量和计算量的增加而可预测地提升。

**(1) 模型架构 (Decoder-Only)**: 

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/3-0.png" alt="alt text" width="100%">
    <p>图3 GPT 模型结构</p>
</div>

仅使用Transformer的**Decoder**部分堆叠而成。其内部的注意力层是**掩码自注意力（Masked Self-Attention）**，确保了生成的自回归特性。



**(2) 预训练任务 (CLM)**: 采用经典的**因果语言模型（Causal Language Model）**，即根据前面所有token，预测下一个token，通过不断重复该过程来实现目标文本序列的生成。例如，CLM 的输入和输出可以是：

    input: 今天天气
    output: 今天天气很
    
    input: 今天天气很
    output：今天天气很好



**(3) GPT系列发展历程**

- **GPT-1 (117M)**: 首次成功验证了“通用预训练+下游任务微调”在生成模型上的可行性。
- **GPT-2 (1.5B)**:
  - **规模剧增**: 模型参数和训练数据量级提升。
  - **Zero-Shot探索**: 展示了在不进行任何微调的情况下，大模型通过理解Prompt就能执行某些任务的能力。
  - **架构微调**: 将LayerNorm移到子层输入之前（**Pre-Norm**），使深层模型训练更稳定。
- **GPT-3 (175B)**:
  - **开启LLM时代**: 其巨大的规模（1750亿参数）带来了**涌现能力（Emergent Abilities）**，即模型在达到一定规模后才表现出的、小模型不具备的复杂能力。
  - **Few-Shot / 上下文学习 (In-context Learning)**: GPT-3的核心突破。通过在Prompt中提供少量（1到几十个）任务示例，模型就能“领悟”并解决该任务，而无需更新任何模型权重。这极大地改变了NLP任务的解决范式。
  - **技术细节**: 使用了交替的稠密和稀疏注意力层来节约计算。



### 3.2 LLaMA系列

LLaMA系列是Meta发布的、奠定了现代开源LLM技术栈基础的模型。

**(1) 模型架构 (Decoder-Only)**: 

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/3-figures/3-1.png" alt="alt text" width="100%">
    <p>图3.13 LLaMA-3 模型结构</p>
</div>
继承并优化了GPT的Decoder-Only架构。

- **关键改进**:
  1. **RMSNorm**: 使用RMSNorm进行预归一化，提升训练稳定性。
  2. **SwiGLU激活函数**: 在FFN层使用SwiGLU替换ReLU，提升了性能。
  3. **旋转位置编码 (RoPE)**: 放弃了绝对/可学习位置编码，改用RoPE。RoPE通过在注意力计算中旋转Query和Key的嵌入向量来注入相对位置信息，具有更好的外推性。



**(2) 发展历程**:

- **LLaMA-1**: 证明了在海量数据上训练的中等规模模型（7B-65B），其性能可以媲美甚至超越规模远大于它的模型（如GPT-3 175B）。
- **LLaMA-2**: 增大了预训练数据量和上下文长度（4k），并首次发布了经过**RLHF**对齐的**Chat**版本。引入了**分组查询注意力 (GQA)**来优化推理速度。
- **LLaMA-3**: 再次大幅增加预训练数据（15T+ tokens），并使用了效率更高的128K词表Tokenizer，在多语言和推理能力上取得显著进步。



### 3.3 GLM系列

GLM系列是智谱AI开发的、在中文LLM领域具有重要影响的模型，其早期版本展示了独特的架构探索。

- **早期模型架构 (GLM-130B, ChatGLM-6B)**:
  - **独特之处**:
    1. **Post-Norm**: 坚持使用Post-Norm，认为其鲁棒性更好。
    2. **激活函数**: 使用**GeLUs**。
    3. **输出层**: 使用单个线性层进行预测。
- **预训练任务 (GLM)**:
  - **核心创新**: 提出**自回归空白填充（Autoregressive Blank Infilling）任务，巧妙地将MLM的自编码思想**和**CLM的自回归思想**相结合。模型既要理解双向上下文来定位被mask的span，又要自回归地生成这个span的内容。
- **GLM家族的发展**:
  - **早期**: ChatGLM-6B作为早期开源中文LLM，降低了研究门槛。
  - **后期 (ChatGLM2及以后)**: 为了追求更强的性能和更好的扩展性，后续版本在架构上**向主流的LLaMA架构（如Pre-Norm, RMSNorm, RoPE）和纯CLM预训练任务靠拢**，这反映了业界对于LLM最佳实践的趋同。
  - **GLM-4**: 最新一代模型，在多项能力上达到SOTA水平，并开源了其强大的轻量级版本。
