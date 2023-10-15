### BERT huggingface 源码阅读：

#### 嵌入层：

###### 源码：

```python
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

```

#### 向前传播部分：

##### 获取input_size：

###### 输入的input_ids：

```python
[[101, 102, 103, 104, 105],  # 句子1的整数ID序列
 [106, 107, 108, 109, 110]]  # 句子2的整数ID序列
```

###### 输入的input_embeddings：

```python
[
    [  # 第一个批次中的句子
        [0.1, 0.2, 0.3, ..., 0.5],  # 第一个句子中第一个token的嵌入向量
        [0.6, 0.7, 0.8, ..., 1.0],  # 第一个句子中第二个token的嵌入向量
        ...
        [0.1, 0.2, 0.3, ..., 0.5]   # 第一个句子中最后一个token的嵌入向量
    ],
    [  # 第二个批次中的句子
        [0.2, 0.3, 0.4, ..., 0.6],  # 第二个句子中第一个token的嵌入向量
        [0.7, 0.8, 0.9, ..., 1.1],  # 第二个句子中第二个token的嵌入向量
        ...
        [0.2, 0.3, 0.4, ..., 0.6]   # 第二个句子中最后一个token的嵌入向量
    ],
    [  # 第三个批次中的句子
        [0.3, 0.4, 0.5, ..., 0.7],  # 第三个句子中第一个token的嵌入向量
        [0.8, 0.9, 1.0, ..., 1.2],  # 第三个句子中第二个token的嵌入向量
        ...
        [0.3, 0.4, 0.5, ..., 0.7]   # 第三个句子中最后一个token的嵌入向量
    ]
]
```

###### `input_ids`的shape应该是(batch_size,max_seq_length)

###### `input_embedding`的shape应该是(batch_size,max_seq_length,embedding_dim)

##### 所以使用：

```python
if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
```

在获取input函数的时候就不需要input_embedding的最后一个维度的信息了，使用`inputs_embeds.size()[:-1]`来把最后一个维度的信息来截取掉，这里获得的 `input_shape` 就是(batch_size,max_seq_length)

---

```python
if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]
```

输入的时候其实是可以自己来输入位置编码的，但是如果你没有传入位置编码的话，程序自动根据你输入的序列长度进行填充位置编码，这里也是相当于huggingface留了一个窗口来传入位置编码。

`past_key_values_length` 的实际运用场景通常是在处理长文本或者序列时，特别是在生成式任务（如文本生成）中。在这些任务中，当处理的文本序列很长时，为了提高模型的效率，可能需要将文本分成多个片段进行处理。

在处理第一个片段时，`past_key_values_length` 通常会被设置为 0，表示从头开始处理。当处理后续的片段时，`past_key_values_length` 的值会被设置为已经处理过的序列长度。

```python
		'''hasattr(self, "token_type_ids") 检查一下self里面有没有token_type_ids这个属性'''
		if token_type_ids is None:
            if hasattr(self, "token_type_ids"): 
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
        '''这一块就是把token_type_ids改成和输入的形状(input_shape)相同'''
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
		'''如果没有token_type_ids这个张量的话就申请一个0矩阵，应该是解决了开源社区上面提出来的issue(上面有注释)'''
```

`token_type_ids`用来标记句子，区分句子A和句子B（一般用0和1区分？）

```python
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
```

检查一下词嵌入输入的时候有没有被计算好，如果没有计算的话，就用self里面的`word_embeddings`函数计算词嵌入。

然后再给`token_type_embeddings`赋值，完成了句子类型标记的嵌入表示。

这两个语句保证了tokens和句子类型都被转换成了嵌入向量。

```python
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```

最后计算embedding的部分，先将输入的句子的词嵌入和句子类型嵌入先互相相加。

然后根据设定的`self.position_embedding_type`来生成位置嵌入，再将位置嵌入也加入到最后的embeddings中去。

然后通过一个`layernorm`层和一个`dropout`层然后输出最后的embeddings

---

### `Bertencoder`部分

```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

```

```python
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
```

这段代码是在为函数的输出预先分配存储空间。具体来说：

`all_hidden_states` 是一个空的元组（`()`），它将用于存储所有层的隐藏状态。如果 `output_hidden_states` 被设置为 `True`，那么模型的每一层的隐藏状态将被添加到这个元组中。如果 `output_hidden_states` 是 `False`，那么这个变量将被设置为 `None`。

`all_self_attentions` 也是一个空的元组，它将用于存储所有层的自注意力权重。如果 `output_attentions` 被设置为 `True`，那么模型的每一层的自注意力权重将被添加到这个元组中。如果 `output_attentions` 是 `False`，那么这个变量将被设置为 `None`。

`all_cross_attentions` 也是一个空的元组，它将用于存储所有层的交叉注意力权重（如果模型支持交叉注意力的话）。这个变量只有在 `output_attentions` 为 `True` 且模型配置中 `add_cross_attention` 为 `True` 时才会被分配空间，否则它将被设置为 `None`。

---

###### python里面这个if else居然是个三目运算符，太震撼了👉👈具体如下

```python
value_if_true if condition else value_if_false
```

---

```python
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
```

检查模型是否启用了梯度检查点（gradient checkpointing）并且是否在训练模式下。梯度检查点是一种技术，它可以减少计算图中需要保存的中间变量的数量，从而减小内存占用。

在梯度检查点模式下，因为计算图被优化，所以缓存可能无法正常工作。

如果打开了梯度检查点而且在训练模式下的话如果使用了缓存就会记录在日志里面，顺便关闭掉缓存。

---

```python
next_decoder_cache = () if use_cache else None
```

如果用户想使用缓存，`next_decoder_cache` 就被初始化为一个空元组，以便在函数的后续运算中存储中间结果。如果用户不使用缓存，`next_decoder_cache` 就被设置为 `None`，表示不需要存储中间结果。这种灵活性允许函数在不同的上下文中以不同的方式运行。

---

###### 后面的整个for循环应该是在记录运行中的中间状态，整个向前传播的过程应该就是for循环在遍历每一个层。

---

```python
self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
```

这里用一个for循环做了一个列表解析，就是创建了一个有`config.num_hidden_layers`个`BertLayer`层的模型。

---

### **`BertLayer`**层源码:

```python
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

```

每一句加一下注释：

```python
class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 从配置中获取参数
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # 创建自注意力层
        self.attention = BertAttention(config)
        # 检查是否为decoder模式
        self.is_decoder = config.is_decoder
        # 检查是否需要跨层注意力
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果添加了跨层注意力但不是decoder模式，抛出错误
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 创建跨层注意力层
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        # 创建前馈神经网络的中间层
        self.intermediate = BertIntermediate(config)
        # 创建前馈神经网络的输出层
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 获取自注意力过去的键值对
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 处理自注意力层
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力的输出
        attention_output = self_attention_outputs[0]

        # 如果是decoder模式，最后一个输出是自注意力的键值对
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，添加自注意力的输出

        cross_attn_present_key_value = None
        # 如果是decoder模式且提供了encoder的隐藏状态
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                # 如果没有跨层注意力层，抛出错误
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 获取跨层注意力的过去的键值对
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 处理跨层注意力层
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取跨层注意力的输出
            attention_output = cross_attention_outputs[0]
            # 如果需要输出注意力权重，添加跨层注意力的输出
            outputs = outputs + cross_attention_outputs[1:-1]

            # 添加跨层注意力的键值对到自注意力的键值对中
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 将attention_output输入前馈神经网络
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是decoder模式，返回注意力的键值对作为最后的输出
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    # 前馈神经网络的chunk处理函数
    def feed_forward_chunk(self, attention_output):
        # 前馈神经网络的中间层处理
        intermediate_output = self.intermediate(attention_output)
        # 前馈神经网络的输出层处理
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
```

这里面用了**`BertAttention`**类**`BertIntermediate`**类**`BertOutput`**类**`apply_chunking_to_forward`**函数

**`BertAttention` 类**: 这个类实现了Bert模型中的自注意力机制

**`BertIntermediate` 类**: 这个类实现了Bert模型中的前馈神经网络的中间层。

**`BertOutput` 类**: 这个类实现了Bert模型中的前馈神经网络的输出层。

**`apply_chunking_to_forward` 函数**: 这个函数用于处理前馈神经网络的输出。

然后一个类一个类的看。

---

先是`BertAttention`类

```python
class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层，传入配置和位置编码类型
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化自注意力的输出层
        self.output = BertSelfOutput(config)
        # 用于存储被剪枝的注意力头的索引
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 查找可剪枝的注意力头
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )
        # 对线性层进行剪枝
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # 更新超参数并存储被剪枝的头的索引
        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 使用自注意力层计算注意力
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用输出层处理注意力输出
        attention_output = self.output(self_outputs[0], hidden_states)
        # 将注意力输出和其他可能的输出（如注意力分数等）一起返回
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要，返回注意力分数
        return outputs

```

然后这里又调用了四个自定义的类和函数

1. **`BertSelfAttention` 类**：这是 `BertAttention` 类中的一个子模块，处理自注意力机制的计算。
2. **`BertSelfOutput` 类**：这是 `BertAttention` 类中的另一个子模块，处理自注意力输出的计算。
3. **`find_pruneable_heads_and_indices` 函数**：该函数用于查找可以被剪枝的注意力头的索引。
4. **`prune_linear_layer` 函数**：该函数用于剪枝线性层（全连接层）的权重。

---

###### `反着读还是有点太痛苦了还是正着一个一个函数看吧`

---

```python
logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    ...
    # See all BERT models at https://huggingface.co/models?filter=bert
]

```

1. `logger = logging.get_logger(__name__)`: 从之前导入的`logging`模块获取当前文件的日志记录器。它可以用于记录信息，警告，错误等。
2. `_CHECKPOINT_FOR_DOC = "bert-base-uncased"`: 定义一个字符串常量，指定文档中要引用的BERT模型的检查点。
3. `_CONFIG_FOR_DOC = "BertConfig"`: 定义一个字符串常量，表示BERT模型的配置名。
4. `# TokenClassification docstring`: 这是一个注释，说明接下来的常量与Token分类任务有关。
5. `_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"`: 定义Token分类任务的BERT模型检查点。
6. `_TOKEN_CLASS_EXPECTED_OUTPUT`: 这是Token分类任务的期望输出，是一个标签序列的示例。
7. `_TOKEN_CLASS_EXPECTED_LOSS = 0.01`: 这是Token分类任务的期望损失。
8. `# QuestionAnswering docstring`: 这是一个注释，说明接下来的常量与问题回答任务有关。
9. `_CHECKPOINT_FOR_QA = "deepset/bert-base-cased-squad2"`: 定义问题回答任务的BERT模型检查点。
10. `_QA_EXPECTED_OUTPUT`: 这是问题回答任务的期望输出，是一个答案的示例。
11. `_QA_EXPECTED_LOSS = 7.41`: 这是问题回答任务的期望损失。
12. `_QA_TARGET_START_INDEX = 14`: 这是答案在文本中的开始索引。
13. `_QA_TARGET_END_INDEX = 15`: 这是答案在文本中的结束索引。
14. `# SequenceClassification docstring`: 这是一个注释，说明接下来的常量与序列分类任务有关。
15. `_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "textattack/bert-base-uncased-yelp-polarity"`: 定义序列分类任务的BERT模型检查点。
16. `_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"`: 这是序列分类任务的期望输出。
17. `_SEQ_CLASS_EXPECTED_LOSS = 0.01`: 这是序列分类任务的期望损失。
18. `BERT_PRETRAINED_MODEL_ARCHIVE_LIST`: 这是一个列表，包含了许多预训练的BERT模型的名称。

这里的期望损失啥的应该就是bert模型的baseline，后续的开发者自己训练模型可以跟baseline进行比较。

---

```python
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re

        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info(f"Converting TensorFlow checkpoint from {tf_path}")
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info(f"Skipping {'/'.join(name)}")
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info(f"Skipping {'/'.join(name)}")
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            if pointer.shape != array.shape:
                raise ValueError(f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched")
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)
    return model
```

这是一个从TensorFlow检查点中加载权重到PyTorch BERT模型的函数。我会逐行解释这个函数的功能。

函数定义：`load_tf_weights_in_bert(model, config, tf_checkpoint_path)`
参数：

- `model`: PyTorch BERT模型。
- `config`: BERT模型的配置信息。
- `tf_checkpoint_path`: TensorFlow检查点的路径。

函数内容：

1. 尝试导入`re`、`numpy`和`tensorflow`库。如果其中任何一个库没有安装，将发出一个错误消息并抛出异常。
2. 使用`os.path.abspath`获取TensorFlow检查点的绝对路径，并打印日志消息。
3. 使用TensorFlow的`tf.train.list_variables`函数列出检查点中的所有变量，并存储其名称和形状。
4. 对于每个变量，加载其权重并将名称和权重数组添加到`names`和`arrays`列表中。
5. 使用`zip`遍历`names`和`arrays`。对于每个名称和数组：
   - 使用`split`方法将名称按`/`分割。
   - 跳过与Adam优化器相关的变量，因为它们在预训练模型中不需要。
   - 然后开始为PyTorch模型指定一个指针，该指针将指向应将权重加载到的位置。
   - 使用循环遍历名称的每个部分，并根据部分名称更新指针位置。
   - 根据名称的最后一部分更新指针。
   - 如果TensorFlow权重的形状与PyTorch模型中的形状不匹配，抛出一个值错误。
   - 最后，使用`torch.from_numpy`将numpy数组转换为PyTorch张量，并将其分配给指针。
6. 返回加载了权重的PyTorch模型。

该函数的主要目的是将TensorFlow格式的预训练BERT模型转换为PyTorch格式。这在迁移学习场景中很有用，尤其是当在PyTorch中使用在TensorFlow中训练的模型时。

---

下面是embedding的部分，上面讲过了，这个函数跳过，下一个

---

### SelfAttention部分

```python
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

```

好的，下面是 `BertSelfAttention` 类的代码段及其解释：

```python
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
```
这里我们检查是否可以将隐藏层大小均匀地分割成多个注意力头。如果不能，我们就抛出一个错误。

```python
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
```
这几行代码定义了注意力头的数量、每个头的大小和所有头的总大小。

```python
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
```
这里我们定义了三个线性层，用于生成“查询”、“键”和“值”向量。这些向量是自注意力机制的核心部分。

```python
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
```
dropout层，用于防止模型过拟合。

```python
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
```
**解释**: 这行代码决定了位置嵌入的类型。默认情况下，它使用“绝对”位置嵌入。

```python
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
```
如果使用相对位置嵌入，我们需要定义一个额外的嵌入层。

```python
        self.is_decoder = config.is_decoder
```
这个变量决定了模块是否用于解码器。如果是，它的行为会有所不同。

---

```python
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
		self.query = nn.Linear(config.hidden_size, self.all_head_size)
    	self.key = nn.Linear(config.hidden_size, self.all_head_size)
    	self.value = nn.Linear(config.hidden_size, self.all_head_size)
```

看这里其实可以解释代码是如何实现多注意力头之间的并行运算的：

![](https://pic3.zhimg.com/80/v2-5e8c504cd21a2b894b1bc5db14fb76be_1440w.webp)

看右边这个图，按照bert原文的设计，一个词向量转化为768维的embedding。每个注意力头是64维度，一共12个头。

![image-20231014195306605](C:\Users\geshasugarsugar\AppData\Roaming\Typora\typora-user-images\image-20231014195306605.png)

这里做的计算单独拿一个embedding出来举例子这里的线性层就是起到了W矩阵的作用，而且是原论文是通过一个W映射到了另一个矩阵，那一个矩阵就是一个Q或一个K或一个V，原先的BERT模型这里使用的是12个维度为64的头，所以这里代码里面用的线性层设置成了

###### nn.Linear(config.hidden_size, self.all_head_size)

###### self.all_head_size = self.num_attention_heads * self.attention_head_size

然后输出的其实就是12份不同注意力的QorKorV

如果要处理一个句子的话就是许多个embedding组成一个句子

然后直接组成一个大的矩阵然后直接乘起来，就实现了多个头的计算的并行

---

接下来的 `transpose_for_scores` 函数和 `forward` 函数包含了模块的主要逻辑。这些函数处理输入数据，执行自注意力运算，然后返回输出。

### transpose_for_scores函数

```python
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
```
这是一个辅助函数，它将输入的张量 `x` 进行重塑和置换，以准备计算注意力得分。

```python
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
```
这行代码创建了一个新的形状 `new_x_shape`。这个形状保留了 `x` 的前面所有维度，并在最后添加了注意力头的数量和每个头的大小。

```python
        x = x.view(new_x_shape)
```
`view` 方法用于重塑张量。这里，我们将 `x` 重塑为新的形状 `new_x_shape`。

```python
        return x.permute(0, 2, 1, 3)
```
`permute` 方法用于置换张量的维度。这里，我们将第二和第三维度互换，使得注意力头位于正确的位置。

### forward函数

```python
    def forward(
        self,
        hidden_states: torch.Tensor,
        ...
    ) -> Tuple[torch.Tensor]:
```
`forward` 函数是模块的核心，负责执行自注意力计算。

```python
        mixed_query_layer = self.query(hidden_states)
```
使用查询（Q）线性层对隐藏状态进行变换，得到查询层。

```python
        is_cross_attention = encoder_hidden_states is not None
```
检查是否进行跨注意力。跨注意力用于模型的解码器部分，其中解码器关注编码器的输出。

以下部分代码根据是否进行跨注意力以及是否有先前的键/值来确定如何获得键和值的层。

```python
        if is_cross_attention and past_key_value is not None:
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
```
1. 如果进行跨注意力且有过去的键/值，我们使用过去的键/值。
2. 如果进行跨注意力但没有过去的键/值，我们使用编码器的隐藏状态来获得键和值。
3. 如果不进行跨注意力但有过去的键/值，我们使用当前的隐藏状态和过去的键/值来获得键和值。
4. 否则，我们只使用当前的隐藏状态来获得键和值。

接下来的代码将处理查询层的变换。

```python
        query_layer = self.transpose_for_scores(mixed_query_layer)
```
使用之前定义的 `transpose_for_scores` 函数重塑和置换查询层。

接下来，代码计算了原始的注意力得分。

```python
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
```
计算查询层和键层之间的点积，得到原始的注意力得分。

后续的代码处理了位置嵌入、注意力掩码、得分到概率的转换、计算上下文层等步骤，这些步骤是自注意力机制的核心部分。

---

```python
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

这个类 `BertSelfOutput` 是 BERT 模型中自注意力机制之后的一个简短的前馈网络，它用于进一步处理注意力输出并与原始的输入（即自注意力之前的输入）进行残差连接。下面我为你详细解释每一行代码：

### BertSelfOutput类初始化函数

```python
def __init__(self, config):
	super().__init__()
	self.dense = nn.Linear(config.hidden_size, config.hidden_size)
```
这是一个线性层，它的输入和输出都有 `config.hidden_size` 的维度。通常在自注意力输出后，我们使用一个全连接层（或称为密集层）来进行线性变换。

```python
    self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
```
```python
def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
	hidden_states = self.dense(hidden_states)
```
自注意力的输出（`hidden_states`）通过全连接层。

```python
    hidden_states = self.dropout(hidden_states)
```
 然后，对全连接层的输出进行 dropout。

```python
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
```
 接下来，我们将 dropout 后的输出与原始的输入（即自注意力之前的输入，`input_tensor`）进行残差连接（简单地将它们相加）。然后，我们对结果进行层归一化。

```python
    return hidden_states
```
最后返回处理后的 `hidden_states`。

---

```python
class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

```

这是 `BertAttention` 类，这个模块结合了自注意力（通过 `BertSelfAttention`）和前馈网络（通过 `BertSelfOutput`）。它首先通过自注意力机制处理输入，然后将这个输出送入前馈网络。下面我将为你详细解释每一部分代码。

### `BertAttention` 类初始化函数

```python
def __init__(self, config, position_embedding_type=None):
    super().__init__()
```
这是 `BertAttention` 类的初始化函数，与之前的模块类似，它首先调用父类 `nn.Module` 的初始化函数。

```python
    self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
```
这里初始化 `BertSelfAttention` 模块，并将其命名为 `self.self`。这个模块用于计算自注意力。

```python
    self.output = BertSelfOutput(config)
```
这里初始化 `BertSelfOutput` 模块，它用于处理自注意力的输出。

```python
    self.pruned_heads = set()
```
这是一个集合，用于存储被修剪（或删除）的注意力头。修剪是一种优化技巧，可以减少模型的大小。

### prune_heads 函数

此函数用于修剪注意力头。在某些情况下，可能希望减少模型的大小或计算需求，这可以通过修剪注意力头来实现。

### forward 函数

```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    ...
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor]:
```
这是模块的核心函数，定义了数据如何通过模块流动。

```python
    self_outputs = self.self(
        hidden_states,
        attention_mask,
        ...
        output_attentions,
    )
```
输入 `hidden_states` 首先通过 `BertSelfAttention` 模块，生成自注意力的输出。

```python
    attention_output = self.output(self_outputs[0], hidden_states)
```
然后，自注意力的输出通过 `BertSelfOutput` 模块。

```python
    outputs = (attention_output,) + self_outputs[1:]
```
这里，我们组合了经过前馈网络处理后的输出和其他可能的输出（例如注意力权重），以形成最终的输出。

`BertAttention` 类是一个组合了自注意力和前馈网络的模块，它首先计算自注意力，然后使用前馈网络进一步处理这个输出。

---

```python
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
```

1. ```
   self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
   ```

   - 这是一个全连接层，它会将`hidden_states`从原始的`hidden_size`转换为`intermediate_size`。
   - 通常，`intermediate_size`是`hidden_size`的4倍（例如，当`hidden_size`=768时，`intermediate_size`=3072）。

2. ```
   if isinstance(config.hidden_act, str): ... else: ...
   ```

   - 这部分代码用于设置激活函数`intermediate_act_fn`。
   - 如果`config.hidden_act`是一个字符串，例如`"gelu"`，则从预定义的激活函数字典`ACT2FN`中获取相应的激活函数。
   - 否则，它假定`config.hidden_act`已经是一个函数，并直接使用它。

### forward函数

这个方法定义了如何从输入的`hidden_states`产生输出。

1. `hidden_states = self.dense(hidden_states)`
   - 使用前面定义的全连接层转换`hidden_states`。
2. `hidden_states = self.intermediate_act_fn(hidden_states)`
   - 将激活函数应用于`hidden_states`。这通常是`gelu`函数，但可以根据`config`中的设置进行更改。
3. `return hidden_states`
   - 返回处理后的`hidden_states`。

总之，`BertIntermediate`类的主要作用是将输入的`hidden_states`通过一个全连接层，并应用一个激活函数，通常是`gelu`。这是Transformer中的Feed Forward Network (FFN)部分。

---

```python
class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
```

1. `self.dense = nn.Linear(config.intermediate_size, config.hidden_size)`
   - 这是一个全连接（线性）层。它会将`hidden_states`从`intermediate_size`（经过`BertIntermediate`处理后的尺寸）转换回原始的`hidden_size`。

2. `self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)`
   - 这是一个层归一化（Layer Normalization）模块，它用于规范化`hidden_states`，确保其均值为0，方差为1，有助于模型的稳定训练。

3. `self.dropout = nn.Dropout(config.hidden_dropout_prob)`
   - 这是一个dropout层，它随机设置输入中的一部分值为0。这是一种正则化技术，有助于防止模型过拟合。

### forward函数:
这个方法定义了如何从输入的`hidden_states`产生输出。

1. `hidden_states = self.dense(hidden_states)`
   - 使用前面定义的全连接层转换`hidden_states`。

2. `hidden_states = self.dropout(hidden_states)`
   - 应用dropout，随机丢弃`hidden_states`中的一些值。

3. `hidden_states = self.LayerNorm(hidden_states + input_tensor)`
   - 将`hidden_states`与`input_tensor`相加。这是一个残差连接，它有助于网络学习恒等映射，从而深化网络而不损失性能。
   - 对结果进行层归一化。

4. `return hidden_states`
   - 返回处理后的`hidden_states`。

总结：`BertOutput`类是BERT的一个重要组件，它定义了如何对从`BertIntermediate`模块接收的`hidden_states`进行处理。处理包括应用一个全连接层、dropout和层归一化，以及与原始的`input_tensor`进行残差连接。

---

```

```

1. 关于注意力机制的设置:

   - `self.attention = BertAttention(config)`：创建一个自注意力机制。

   - `self.is_decoder = config.is_decoder`：一个布尔标志，表示是否作为解码器使用这个模块。

   - ```
     self.add_cross_attention = config.add_cross_attention
     ```

     ：一个布尔标志，表示是否应该添加交叉注意力。

     - 如果添加了交叉注意力，那么这个模块应该用作解码器模型，否则会抛出错误。交叉注意力用于编码器-解码器结构中的解码器部分，使得解码器可以注意到编码器的输出。

2. 关于前馈神经网络的设置:

   - `self.intermediate = BertIntermediate(config)`：用于对注意力的输出进行进一步转换。
   - `self.output = BertOutput(config)`：用于从中间输出产生最终的层输出。

### `BertLayer` 的前向传播方法 (`forward`):

1. **自注意力的应用**:
   - 使用`self.attention`模块处理`hidden_states`，并获得注意力的输出。
2. **(可选) 交叉注意力的应用**:
   - 如果这个模块被用作一个解码器并且有编码器的`hidden_states`传入，那么会使用交叉注意力模块处理注意力的输出。
3. **应用前馈神经网络**:
   - `layer_output = apply_chunking_to_forward(self.feed_forward_chunk, ...)`：在`attention_output`上应用前馈神经网络。这里使用了`apply_chunking_to_forward`函数，它可以分块地应用前馈函数，以减少内存使用量。
   - `self.feed_forward_chunk`是一个辅助函数，它处理每一块的前馈操作。

### `feed_forward_chunk` 方法:

这个方法简单地应用前馈神经网络到`attention_output`。

1. `intermediate_output = self.intermediate(attention_output)`：首先，应用中间转换。
2. `layer_output = self.output(intermediate_output, attention_output)`：然后，使用`BertOutput`模块产生最终的层输出。

总结：`BertLayer`类代表BERT模型中的一个单独的层。每个`BertLayer`都包括一个多头自注意力子层、(可选的)一个多头交叉注意力子层和一个前馈神经网络子层。这三个子层都带有残差连接和层归一化。

---

```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
```

### 初始化方法 (`__init__`):

1. `self.config = config`：存储模型的配置。
2. `self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])`：创建一个模块列表，包含多个`BertLayer`。数量由`config.num_hidden_layers`确定。
3. `self.gradient_checkpointing = False`：一个标志，表示是否使用梯度检查点以减少内存使用。

### `forward`函数:

1. **初始化输出的存储**:
   - `all_hidden_states`, `all_self_attentions`, 和 `all_cross_attentions` 被初始化为元组，用于存储每一层的输出。
2. **梯度检查点设置**:
   - 如果使用了梯度检查点和缓存，则会发出警告。
3. **遍历每个`BertLayer`**:
   - 对于每一个`BertLayer`，先检查是否要输出隐藏状态。
   - 设置特定于该层的`head_mask`和`past_key_value`。
   - 使用梯度检查点或直接调用该层的`forward`方法，处理`hidden_states`。
4. **收集和返回输出**:
   - 如果`output_hidden_states`为True，将最后一层的输出添加到`all_hidden_states`。
   - 根据`return_dict`的值，返回一个元组或一个命名元组。

简而言之，`BertEncoder`类通过遍历每个`BertLayer`来处理输入的`hidden_states`。每个`BertLayer`都提供一个新的`hidden_states`，这个新的`hidden_states`在下一个`BertLayer`中作为输入。在所有层之后，最后的`hidden_states`（以及其他可选输出）被返回。

---

```
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

### `BertPooler` 类的初始化方法 (`__init__`):

1. `self.dense = nn.Linear(config.hidden_size, config.hidden_size)`：定义一个线性层，它将输入大小和输出大小都设置为`config.hidden_size`。
2. `self.activation = nn.Tanh()`：定义一个双曲正切激活函数。

### `BertPooler` 的前向传播方法 (`forward`):

1. `first_token_tensor = hidden_states[:, 0]`：从输入的`hidden_states`中选择第一个token的所有隐藏状态。在BERT中，第一个token通常是特殊的`[CLS]` token。
2. `pooled_output = self.dense(first_token_tensor)`：将选择的第一个token的隐藏状态传递给前面定义的线性层。
3. `pooled_output = self.activation(pooled_output)`：将线性层的输出传递给双曲正切激活函数，得到`pooled_output`。
4. `return pooled_output`：返回得到的`pooled_output`。

总之，`BertPooler`的任务是从输入的`hidden_states`中选取第一个token的隐藏状态，然后通过一个线性层和一个激活函数，得到一个固定大小的输出向量。这个输出向量常常被用作分类任务的表示。

---

看到这里整个huggingface里面是怎么实现这个bert就比较清楚了，也具体的讲了他是咋实现多头注意力机制的

这个整个的实现基本上的实现和修改基本上通过修改config就可以做到比较好的自己调整整个bert模型，在程序的一开始里面的参数里也告诉了我们这个模型的baseline是多少。

## 再之后的函数就是各类任务的转化头了，Bert可以用来完成各类任务（应该就是对应之前那本书里面的模块L）

---

```python
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
```

这是BERT模型中的预测头转换部分，用于BERT的Masked Language Model (MLM)任务中，对每个位置的隐藏状态进行进一步的转换，从而为下一步的预测任务做准备。

### 初始化方法 (`__init__`):

1. `self.dense = nn.Linear(config.hidden_size, config.hidden_size)`：定义一个线性层，输入和输出大小都是`config.hidden_size`。
2. `if isinstance(config.hidden_act, str):`：这个条件检查`config.hidden_act`是否是字符串类型。
   - `self.transform_act_fn = ACT2FN[config.hidden_act]`：如果`config.hidden_act`是字符串，那么它从一个预先定义的字典`ACT2FN`中选择对应的激活函数。
   - `else: self.transform_act_fn = config.hidden_act`：如果不是字符串，则直接使用`config.hidden_act`作为激活函数。
3. `self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)`：定义一个层归一化（LayerNorm），它对输入的每一层进行归一化。

### `forward`函数:

1. `hidden_states = self.dense(hidden_states)`：将输入的`hidden_states`传递给定义的线性层。
2. `hidden_states = self.transform_act_fn(hidden_states)`：将线性层的输出传递给选定的激活函数。
3. `hidden_states = self.LayerNorm(hidden_states)`：使用层归一化处理激活函数的输出。
4. `return hidden_states`：返回处理后的`hidden_states`。

总的来说，`BertPredictionHeadTransform`的任务是对输入的`hidden_states`进行进一步的转换，包括一个线性层、一个激活函数和一个层归一化，从而得到一个新的隐藏状态，为下一步的预测任务做准备。

---

```python
class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
```

这个类`BertLMPredictionHead`是BERT的语言模型预测头，用于在Masked Language Modeling (MLM)任务中为每个位置的token进行预测。它基于隐藏状态为每个token位置生成词汇表大小的分数，最高的分数代表模型的预测token。

### 初始化方法 (`__init__`):

1. `self.transform = BertPredictionHeadTransform(config)`：定义一个转换层，它接收隐藏状态并进行进一步的转换（如上面的`BertPredictionHeadTransform`类所示）。
2. `self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)`：定义一个线性层，它接收`transform`层的输出，并将其转换为大小为`config.vocab_size`的输出，代表每个token的分数。注意，这里的偏置被显式地设置为`False`。
3. `self.bias = nn.Parameter(torch.zeros(config.vocab_size))`：定义一个偏置参数，它的大小等于词汇表的大小。
4. `self.decoder.bias = self.bias`：将前面定义的偏置参数与解码器的偏置链接起来。

### `forward`函数:

1. `hidden_states = self.transform(hidden_states)`：首先，将输入的`hidden_states`传递给`transform`层进行转换。
2. `hidden_states = self.decoder(hidden_states)`：接着，将转换后的`hidden_states`传递给解码器，得到每个token的分数。
3. `return hidden_states`：返回这些分数。

总的来说，`BertLMPredictionHead`的任务是接收一个隐藏状态，对其进行转换，然后使用一个解码器生成每个token位置的分数。这些分数后续用于计算损失函数和生成模型的预测。

---

```python
class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
```

用于MLM任务。它包含BERT的语言模型预测头，并用于为给定的输入序列输出预测分数。

### 初始化方法 (`__init__`):

1. `self.predictions = BertLMPredictionHead(config)`：初始化BERT的语言模型预测头，这将用于为输入序列生成预测分数。

### `forward`函数:

1. `prediction_scores = self.predictions(sequence_output)`：将输入的`sequence_output`传递给语言模型预测头，得到预测分数。
2. `return prediction_scores`：返回这些分数。

总的来说，这个模块的作用非常简单：接收一个序列输出，并返回对于这个序列中每个位置的token的预测分数。这些分数通常用于MLM任务，例如预测被遮挡的token。

---

```python
class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
```

这是一个用于Next Sentence Prediction (NSP)任务的模块。在BERT的预训练中，除了Masked Language Modeling (MLM)任务，还有一个Next Sentence Prediction (NSP)任务，用于预测第二句是否在原文中紧随第一句之后。

### 初始化方法 (`__init__`):

1. `self.seq_relationship = nn.Linear(config.hidden_size, 2)`：
   这里定义了一个线性层，其输入维度是`config.hidden_size`（通常是BERT模型的隐藏层维度，如768），输出维度为2。

### `forward`函数:

1. `seq_relationship_score = self.seq_relationship(pooled_output)`：
   使用上述定义的线性层`seq_relationship`对输入的`pooled_output`进行计算。`pooled_output`通常是来自`BertPooler`的输出，表示整个输入序列的固定长度的表示。

2. `return seq_relationship_score`：返回计算得到的得分。这个得分将用于判断第二句是否是第一句的下一句。

这个模块从输入的`pooled_output`中得到一个二分类的得分，这个得分用于预测两句话之间的关系：第二句是否紧随第一句出现。

---

```python
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
```

用于BERT的预训练任务。

BERT的预训练涉及两个主要任务：掩码语言建模（MLM）和下一个句子预测（NSP）。这两个任务的目标是使模型能够学习词嵌入和句子结构。

### 构造函数 (`__init__`):

1. `super().__init__()`：调用`nn.Module`的构造函数。
2. `self.predictions = BertLMPredictionHead(config)`：这是用于MLM任务的头部。它预测每个位置的输出词汇。
3. `self.seq_relationship = nn.Linear(config.hidden_size, 2)`：这是用于NSP任务的线性层。它预测两个句子是否连续。

### `forward`函数:

这是模型的前向传播方法。它的参数是`sequence_output`（来自BERT的每个位置的输出）和`pooled_output`（来自BERT的第一个令牌的输出，通常是`[CLS]`令牌）。

1. `prediction_scores = self.predictions(sequence_output)`：计算每个位置的预测分数，用于MLM任务。
2. `seq_relationship_score = self.seq_relationship(pooled_output)`：计算两个句子是否连续的得分，用于NSP任务。
3. 返回`prediction_scores`和`seq_relationship_score`。

总之，`BertPreTrainingHeads`类是用于BERT预训练的两个任务的组合。它包含两个头部，一个用于预测掩码位置的词，另一个用于预测两个句子是否连续。

---

```
class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value
```

一个抽象类，用于处理权重的初始化，以及为下载和加载预训练模型提供一个简单的接口。这个类继承了`PreTrainedModel`，这意味着它将从`PreTrainedModel`那里继承一些通用的功能和属性。

### 类变量:

1. `config_class = BertConfig`:
   指定与此模型关联的配置类。在这种情况下，它是`BertConfig`。

2. `load_tf_weights = load_tf_weights_in_bert`:
   指定用于从TensorFlow模型加载权重到PyTorch模型的函数。

3. `base_model_prefix = "bert"`:
   模型的前缀名称。通常用于保存和加载模型。

4. `supports_gradient_checkpointing = True`:
   表示该模型支持梯度检查点，这有助于在有限的GPU内存上训练大型模型。

### `_init_weights` 方法:

这个方法是用于初始化模型权重的。根据传入的模块类型，它执行不同的权重初始化。

1. 如果模块是`nn.Linear`，它将使用正态分布初始化权重，并将偏置设置为零。
2. 如果模块是`nn.Embedding`，它也使用正态分布初始化权重，并将任何填充索引的权重设置为零。
3. 如果模块是`nn.LayerNorm`，它将偏置设置为零，权重全部填充为1。

### `_set_gradient_checkpointing` 方法:

这个方法设置给定模块是否应使用梯度检查点。在这里，它特定于`BertEncoder`模块，只有当模块是`BertEncoder`类型时，才会设置其`gradient_checkpointing`属性。

这个类主要是为了提供一个标准的方式来初始化BERT模型的权重，并为更复杂的BERT模型提供一些通用的功能和属性。

---

```python
@dataclass
class BertForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].

    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


BERT_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`BertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
```

这里就是给我们一个文档告诉我们这里的各种各样的属性：

### 属性：

1. `loss`：只有在提供了标签时才返回的可选属性，表示模型的总损失。它是masked语言建模（MLM）损失和下一个序列预测（NSP）损失的总和。
2. `prediction_logits`：这是MLM任务的预测分数。在应用SoftMax函数之前，为每个词汇标记提供分数。
3. `seq_relationship_logits`：这是NSP任务的预测分数。在应用SoftMax函数之前，为序列继续预测（True/False）提供分数。
4. `hidden_states`：只有当 `output_hidden_states` 标志设置为 `True` 时才返回的可选属性，它包含模型在每一层的输出以及初始嵌入输出的隐藏状态。
5. `attentions`：只有当 `output_attentions` 标志设置为 `True` 时才返回的可选属性，它包含了注意力softmax后的注意力权重。

### 文档字符串：

1. `BERT_START_DOCSTRING`：这个文档字符串提供了BERT模型的概述，提到了它从 `PreTrainedModel` 继承，作为一个PyTorch模块的行为，以及如何用配置初始化它。
2. `BERT_INPUTS_DOCSTRING`：这个文档字符串描述了BERT模型的输入参数，解释了每个参数的作用和形状。

---

好了现在可以看整个BERT模型了，也很清晰明了了；

再把上面那些复制下来；

```python
class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )
```

### 构造函数 (`__init__`):

1. `super().__init__(config)`：调用基类的构造函数。
2. `self.config = config`：保存配置。
3. `self.embeddings = BertEmbeddings(config)`：BERT的词嵌入层。
4. `self.encoder = BertEncoder(config)`：BERT的编码器，包含所有的Transformer层。
5. `self.pooler = BertPooler(config) if add_pooling_layer else None`：池化层，如果`add_pooling_layer`为True，则添加。

### `get_input_embeddings` & `set_input_embeddings`:

这两个方法用于获取和设置模型的词嵌入。

### `_prune_heads`:

这个方法用于修剪注意力头。它遍历要修剪的所有层和头，并调用相应层的`prune_heads`方法。

### `forward`:

这是模型的前向传播方法，它定义了模型如何从输入数据生成输出。

1. 它首先检查输入的类型和形状，然后根据需要生成或使用给定的`attention_mask`和`token_type_ids`。
2. 使用`self.embeddings`计算嵌入输出。
3. 使用`self.encoder`计算编码器的输出。
4. 使用`self.pooler`计算池化输出（如果存在）。
5. 根据`return_dict`的值，决定返回一个元组还是一个命名元组。
