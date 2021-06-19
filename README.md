# sohu_text_matching
2021搜狐校园文本匹配算法大赛 分比我们低的都是帅哥队

本repo包含了本次大赛决赛环节提交的代码文件及答辩PPT，提交的模型文件可在百度网盘获取（链接：https://pan.baidu.com/s/1T9FtwiGFZhuC8qqwXKZSNA ，提取码：2333 ）。

最终提交的5个模型（限制大小在2G内）在复赛测试集上的f1指标为0.78921，在决赛测试集上的f1指标为0.78123，在十组队伍中位列第二，最终取得亚军成绩。

复现复赛测试集结果，可将模型下载后放至`checkpoints/rematch`内，将测试集合并为决赛格式后，进入`src`文件夹运行`infer_final.py`并指定输入文件及输出位置即可。依赖项可参考dockerfile。
