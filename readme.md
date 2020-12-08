实现了 BERT 情感二分类  

需要下载 pytorch 版本的 robert_wwm_ext，  
在 model/bert/ 建立文件夹 chinese_roberta_wwm_ext_pytorch  
然后将下载的预训练模型放入其中

model/bert/chinese_roberta_wwm_ext_pytorch

+ config.json
+ pytorch_model.bin  

***

程序入口：run.py   
设置 do_prediction 来训练或测试  

使用多块 GPU：
如设置 gpu_ids = "1,2,3"

测试准确率：91%

***

环境：pytorch 1.5.0 + cuda10.1,  

python3.6.9, transformers 3.2.0


