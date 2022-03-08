分类器：

运行环境：
python == 3.6.13
torch == 1.9.0 (pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==1.9.0)
transformers == 2.9.0 (pip install -i https://pypi.tuna.tsinghua.edu.cn/simple transformers==2.9.0)
cudnn (如果没有安装cuda)
pandas == 1.1.5
numpy == 1.19.5
pyyaml

运行准备：
1、新建文件夹bert-base-chinese，下载链接中的内容放入该文件夹
    bert-base-chinese：
    链接：https://pan.baidu.com/s/1t6SeXown9HenSWAdS9JzIQ 
    提取码：p9f7
2、新建文件夹model，用于之后存储模型
3、新建文件夹preprocess，用于存储label_table
4、新建文件夹logger，用于存储日志

基本思路：
使用transformer中的中文分词工具对输入的中文文本进行处理，然后放入线性分类器进行训练

一、主题识别
模型训练：
python main.py

使用训练好的模型做主题识别：
python theme_recognize.py

二、意图识别
模型训练：
python main.py --task='intent'

使用训练好的模型做意图识别：
python intent_recognize.py

三、情绪识别
模型训练：
python main.py --task='sentiment'

使用训练好的模型做情绪识别：
python sentiment_recognize.py
