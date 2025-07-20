# Happy-LLM 学习笔记 Task 03: Transformer架构

> **课程来源**: [Datawhale Happy-LLM 官网](https://datawhalechina.github.io/happy-llm/)

> **我的笔记仓库**: [https://github.com/GuYan66/happy-llm.git](https://github.com/GuYan66/happy-llm.git)

---



《Attention is all you need》论文：[Attention is all you need](https://arxiv.org/pdf/1706.03762)



## Seq2Seq模型与Transformer整体架构

### Seq2Seq模型

序列到序列（Seq2Seq）是最经典的NLP任务，指模型输入一个自然语言序列 $input = (x_1, x_2, x_3...x_n)$ ，输出的是一个可能不等长的自然语言序列 $output = (y_1, y_2, y_3...y_m)$ 。几乎所有NLP 任务都可以视为 Seq2Seq 任务。

对于 Seq2Seq 任务，一般的思路是对自然语言序列进行编码再解码。

**编码 (Encode)**: 就是将输入的自然语言序列通过隐藏层编码成能够表征语义的向量（或矩阵），可以简单理解为更复杂的词向量表示。

**解码 (Decode)**: 就是对输入的自然语言序列编码得到的向量或矩阵通过隐藏层输出，再解码成对应的自然语言目标序列。



### Transformer 整体架构

Transformer 中的 **Encoder**，就是用于上述的编码过程；**Decoder** 则用于上述的解码过程。

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/2-0.jpg" alt="图片描述" width="60%"/>
  <p>编码器-解码器结构</p>
</div>


Transformer 由 Encoder 和 Decoder 组成，每一个 Encoder（Decoder）又由 6个 Encoder（Decoder）Layer 组成。输入源序列会进入 Encoder 进行编码，到 Encoder Layer 的最顶层再将编码结果输出给 Decoder Layer 的每一层，通过 Decoder 解码后就可以得到输出目标序列了。



## Transformer 核心组件详解

单独挑出一个Encoder细看，可以分为三个部分：**输入部分**、**注意力机制**、**前馈神经网络**。此外还包含**层归一化**和**残差连接**这两个关键的辅助组件。

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/1.png" alt="图片描述" width="70%"/>
</div>



### 输入部分

#### Embedding层

它是一个存储固定大小的词典的嵌入向量查找表。让自然语言输入通过分词器 tokenizer，分词器的作用是把自然语言输入切分成 token 并转化成一个固定的 index。

因此，Embedding 层的输入往往是一个形状为 （batch_size，seq_len，1）的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer 转化成的 index 值。例如，对上述输入，Embedding 层的输入会是：

```
[[[0],[1],[2]]]
```

其 batch_size 为1，seq_len 为3，转化出来的 index 如上。

而 Embedding 内部其实是一个可训练的`（Vocab_size，embedding_dim）`的权重矩阵，词表里的每一个值，都对应一行维度为 `embedding_dim` 的向量。对于输入的值，会对应到这个词向量，然后拼接成`（batch_size，seq_len，embedding_dim）`的矩阵输出。



#### 位置编码

因为后面的多头注意力机制是并行化处理文字，所以需要位置编码来对文字的位置信息进行记录。

令位置为 `pos`，嵌入维度为 d_model，则位置编码定义如下（偶数维度使用sin，奇数维度使用cos）：
$$
PE_{(pos,\, 2i)} = \sin\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

$$
PE_{(pos,\, 2i+1)} = \cos\left(\frac{pos}{10000^{\frac{2i}{d_{model}}}}\right)
$$

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/2.png" alt="图片描述" width="60%"/>
</div>

再将位置编码的512个维度和词向量的512个维度**相加**得到最终的512个维度作为整个Transformer的输入。

**为什么位置嵌入是有用的？** 

这种正余弦编码方式的一个重要特性是，绝对位置向量中蕴含了相对位置信息。也就是说，任意位置 `pos+k` 的位置编码都可以被 `pos` 位置的编码线性表示，这使得模型能轻易学习到相对位置关系。



### 注意力机制

注意力机制的三个核心变量：查询值 Query，键值 Key 和 真值 Value。
$$
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$
接下来一步步的推导上面完整的注意力机制的公式

点积可以度量词向量的相似度，越相似值就越大。可以计算 Query 和每一个键的相似程度：
$$
x = qK^T
$$

此处的 K 即为将所有 Key 对应的词向量堆叠形成的矩阵。基于矩阵乘法的定义，x 即为 q 与每一个 k 值的点积。现在我们得到的 x 即反映了 Query 和每一个 Key 的相似程度，我们再通过一个 Softmax 层将其转化为和为 1 的权重：

$$
\text{softmax}(x)_i = \frac{e^{xi}}{\sum_{j}e^{x_j}}
$$

这样，得到的向量就能够反映 Query 和每一个 Key 的相似程度，同时又相加权重为 1，也就是我们的注意力分数了。最后，我们再将得到的注意力分数和值向量做对应乘积即可。根据上述过程，我们就可以得到注意力机制计算的基本公式：

$$
attention(Q,K,V) = softmax(qK^T)v
$$

不过，此时的值还是一个标量，同时，我们此次只查询了一个 Query。我们可以将值转化为维度为 $d_v$ 的向量，同时一次性查询多个 Query，同样将多个 Query 对应的词向量堆叠在一起形成矩阵 Q，得到公式：

$$
attention(Q,K,V) = softmax(QK^T)V
$$

目前，我们离标准的注意力机制公式还差最后一步。在上一个公式中，如果 Q 和 K 对应的维度 $d_k$ 比较大，softmax 放缩时就非常容易受影响，使不同值之间的差异较大，从而影响梯度的稳定性。因此，我们要将 Q 和 K 乘积的结果做一个放缩：

$$
attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

这也就是注意力机制的核心计算公式了。

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/3.png" alt="图片描述" width="60%"/>
</div>

Q，K，V是如何得到的呢

X1乘以Wq，Wk，Wv三个参数矩阵分别得到q1，k1，v1；X2乘以Wq，Wk，Wv三个参数矩阵分别得到q2，k2，v2。这里使用的都是同一套矩阵参数

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/4.png" alt="图片描述" width="60%"/>
</div>

接下来计算QK相似度，得到attention值

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/5.png" alt="图片描述" width="60%"/>
</div>

实际代码中使用矩阵，方便并行



#### 自注意力

Transformer的Encoder结构中，使用的是自注意力（self-attention）机制。即是计算本身序列中每个元素对其他元素的注意力分布，即在计算过程中，**Q、K、V 都由同一个输入序列通过不同的参数矩阵计算得到**。在 Encoder 中，Q、K、V 分别是输入对参数矩阵$W_q、W_k、W_v$做积得到，从而拟合输入语句中每一个 token 对其他所有 token 的关系。



#### 掩码自注意力

**动机**: 传统的串行预测方式效率低。Transformer使用掩码机制来实现并行计算，同时防止模型在训练时“看到”未来的信息。

掩码自注意力（Mask Self-Attention）指使用注意力掩码的自注意力机制。它会生成一串掩码，来遮蔽未来信息。例如，我们待学习的文本序列仍然是 【BOS】I like you【EOS】，我们使用的注意力掩码是【MASK】，那么模型的输入为：

    <BOS> 【MASK】【MASK】【MASK】【MASK】
    <BOS>    I   【MASK】 【MASK】【MASK】
    <BOS>    I     like  【MASK】【MASK】
    <BOS>    I     like    you  【MASK】
    <BOS>    I     like    you   </EOS>

在每一行输入中，模型仍然是只看到前面的 token，预测下一个 token。但是注意，上述输入不再是串行的过程，而可以一起并行地输入到模型中，模型只需要每一个样本根据未被遮蔽的 token 来预测下一个 token 即可，从而实现了并行的语言模型。



#### 多头注意力

**动机**: 单一的注意力计算只能拟合一种相关关系。多头机制允许模型从不同角度（不同的表示子空间）去理解文本，捕捉更丰富的语义信息。

多头注意力机制（Multi-Head Attention），即同时对一个语料进行多次注意力计算，每次注意力计算都能拟合不同的关系，将最后的多次结果拼接起来作为最后的输出，即可更全面深入地拟合语言信息，用一张很形象的图来说明。

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/6.png" alt="图片描述" width="80%"/>
  <p>多头注意力</p>
</div>

**机制：**有多套独立的 $W_q$, $W_k$, $W_v$ 权重矩阵，每一套矩阵构成一个“头”。将输入并行地送入每个头进行注意力计算，得到多个输出，之后将这些输出拼接起来，再通过一个线性层进行融合。

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/7.png" alt="图片描述" width="70%"/>
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/8.png" alt="图片描述" width="70%"/>
</div>



### 残差连接

Transformer 采用了残差连接的思想来连接每一个子层。残差连接，即下一层的输入不仅是上一层的输出，还包括上一层的输入。例如，在 Encoder 中，在第一个子层，输入进入多头自注意力层的同时会直接传递到该层的输出，然后该层的输出会与原输入相加，再进行标准化。在第二个子层也是一样。即：
$$
x = x + MultiHeadSelfAttention(LayerNorm(x))
$$

$$
output = x + FNN(LayerNorm(x))
$$

<div align="center">
  <img src="https://raw.githubusercontent.com/GuYan66/happy-llm/main/images/2-figures/9.png" alt="图片描述" width="60%"/>
  <p>残差和LayNorm</p>
</div>

残差网络可以有效缓解深度网络中的**梯度消失**问题，让信息和梯度能更顺畅地在网络中流动，从而使训练更深的模型成为可能。



### 层归一化

神经网络主流的归一化有两种，批归一化（Batch Norm）和层归一化（Layer Norm）。Transformer使用的是**层归一化**。

下面介绍批归一化，批归一化是指在一个 mini-batch 上进行归一化，首先计算均值：
$$
\mu_j = \frac{1}{m}\sum^{m}_{i=1}Z_j^{i}
$$

再计算样本的方差：

$$
\sigma^2 = \frac{1}{m}\sum^{m}_{i=1}(Z_j^i - \mu_j)^2
$$

最后，对每个样本的值减去均值再除以标准差来将这一个 mini-batch 的样本的分布转化为标准正态分布：

$$
\widetilde{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \epsilon}}
$$

LN是针对同一个样本的所有单词去做缩放，BN是对第一个单词做矩阵方差，第二个单词做矩阵方差...

Batch Norm 在每一层统计所有样本的均值和方差，Layer Norm 在每个样本上计算其所有层的均值和方差，从而使每个样本的分布达到稳定。Layer Norm 的归一化方式其实和 Batch Norm 是完全一样的，只是统计统计量的维度不同。

**为什么用LN不用BN**: BN的效果依赖于批次大小，且不适合处理长度可变的NLP序列。而LN针对单个样本操作，与批次大小和序列长度无关，在NLP任务中表现更稳定。





## 完整的Transformer模型

最后上述所有组件按照下图结构拼接起来得到完整的Transformer模型

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/3-1.png" alt="图片描述" width="60%"/>
  <p>Transformer 模型结构</p>
</div>
- **Pre-Norm vs Post-Norm**: 需要注意的是，上图是原论文配图，LayerNorm 层放在了 Attention 层**后面**，也就是“Post-Norm”结构。但在其发布的源代码和后续研究中，发现将LayerNorm 层放在 Attention 层**前面**的“Pre-Norm”结构，可以使训练过程的loss更稳定。当前LLM普遍采用“Pre-Norm”结构。
- **整体流程**: 经过 tokenizer 映射后的输出先经过 **Embedding 层**和 **Positional Encoding** 层编码，然后进入N个 **Encoder Layer** 和 N个 **Decoder Layer**（在 Transformer 原模型中，N 取为6），最后经过一个**线性层**和一个 **Softmax 层**就得到了最终输出。
