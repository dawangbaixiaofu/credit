# Embedding Feature
reference: https://github.com/mistralai/mistral-src/blob/main/tutorials/classifier.ipynb


使用开源大模型给传统的模型添加特征的步骤：

- 选择开源的大模型
- 训练集，字段类型为文本
- 使用其tokenizer对文本进行分词到token，转化为id
- 输入id对应的tensor给到大模型进行前馈传递到embedding层
- 大模型输出的结果作为特征
- 对（embedding_feature，label）进行训练/测试切分
- 对embedding_feature进行标准化处理
- 训练传统的分类模型

在把embedding feature给到传统模型建模前，可以对feature进行降低到二维。降维方法可以使用manifold learning方法，画图可视化数据，初步观察是否具有可分性，进行初步判断。
