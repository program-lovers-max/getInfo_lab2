import glob
import os
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
import torch
import torch.nn as nn
from torchcrf import CRF
from seqeval.metrics import accuracy_score as seq_accuracy_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import precision_score as seq_precision_score
from seqeval.metrics import recall_score as seq_recall_score

# from sklearn.metrics import accuracy_score as sklearn_accuracy_score
# from sklearn.metrics import f1_score as sklearn_f1_score
# from sklearn.metrics import precision_score as sklearn_precision_score
# from sklearn.metrics import recall_score as sklearn_recall_score

def read_data(file):
    with open(file,"r",encoding="utf-8") as f:
        all_data = f.read().split("\n")

    all_text = []
    all_label = []

    text = []
    label = []
    for data in all_data:
        if data == "":
            all_text.append(text)
            all_label.append(label)
            text = []
            label = []
        else:
            t,l = data.split(" ")
            text.append(t)
            label.append(l)

    return all_text,all_label

def build_label(train_label):
    label_2_index = {"PAD":0,"UNK":1}  #补充标签
    for label in train_label:
        for l in label:
            if l not in label_2_index:
                label_2_index[l] = len(label_2_index)
    return label_2_index,list(label_2_index)

class BertDataset(Dataset):
    def __init__(self,all_text,all_label,label_2_index,max_len,tokenizer,is_test=True):
        self.all_text = all_text  #所有文本
        self.all_label = all_label  #所有标签
        self.label_2_index = label_2_index  #标签索引字典
        self.tokenizer = tokenizer  #编码器
        self.max_len = max_len   #文本最大长度
        self.is_test = is_test   #是否为测试

    def __getitem__(self,index):
        if self.is_test:   #若是测试集，不截断
            self.max_len = len(self.all_label[index])
        #取出文本和标签
        text = self.all_text[index]
        label = self.all_label[index][:self.max_len]
        #对文本和标签进行编码，注意填充
        text_index = self.tokenizer.encode(text,add_special_tokens=True,max_length=self.max_len+2,padding="max_length",truncation=True,return_tensors="pt")
        label_index = [0] +  [self.label_2_index.get(l,1) for l in label] + [0] + [0] * (self.max_len - len(text))
        #转为tensor格式
        label_index = torch.tensor(label_index)
        #返回索引及长度信息以供模型使用
        return  text_index.reshape(-1),label_index,len(label)

    def __len__(self):
        return self.all_text.__len__()


class Bert_LSTM_NerModel(nn.Module):
    def __init__(self,lstm_hidden,class_num):
        super().__init__()   #父类初始化

        self.bert = BertModel.from_pretrained("bert_base_chinese")  #引用模型
        for name,param in self.bert.named_parameters(): #减少特征，便于快速训练
            param.requires_grad = False
        #引入层数为1的LSTM
        self.lstm = nn.LSTM(768,lstm_hidden,batch_first=True,num_layers=1,bidirectional=False) # 768 * lstm_hidden
        #线性层作为分类器
        self.classifier = nn.Linear(lstm_hidden,class_num)
        #条件随机场，第一维度为batch
        self.crf = CRF(class_num,batch_first=True)
        #损失，有了crf后不再需要
        # self.loss_fun = nn.CrossEntropyLoss()

    def forward(self,batch_index,batch_label=None):
        bert_out = self.bert(batch_index)   #特征
        bert_out0,bert_out1 = bert_out[0],bert_out[1]# bert_out0:字符级别特征, bert_out1:篇章级别
        #经过lstm层
        lstm_out,_ = self.lstm(bert_out0)
        #经过线性分类器
        pre = self.classifier(lstm_out)
        #如果是训练，则返回损失，否则返回crf解码后的预测值
        if batch_label is not None:
            # loss = self.loss_fun(pre.reshape(-1,pre.shape[-1]),batch_label.reshape(-1))
            loss = -self.crf(pre, batch_label)
            return loss
        else:
            return self.crf.decode(pre)



if __name__ == "__main__":


    train_text, train_label = read_data(os.path.join("data", "train.txt")) #导入训练文件
    dev_text,dev_label = read_data(os.path.join("data", "dev.txt")) #导入验证文件
    test_text,test_label = read_data(os.path.join("data", "test.txt")) #导入测试文件

    label_2_index,index_2_label = build_label(train_label)

    tokenizer = BertTokenizer.from_pretrained("bert_base_chinese")  #使用目标编码器
    #设置训练参数
    batch_size = 50  #batch大小
    epoch = 100   #轮次
    max_len = 30  #训练时文本最大长度
    lr = 0.0005  #学习率
    lstm_hidden = 128  #lstm隐藏层大小
    #模式选择
    do_train = False
    do_test = False
    do_input = False
    print('请选择模式:')
    print('是否训练? (需要训练则输入True，否则输入其他字符)')
    tmp = input()
    if tmp == 'True':
        do_train = True
    print('是否测试? (需要测试则输入True，否则输入其他字符)')
    tmp = input()
    if tmp == 'True':
        do_test = True
    print('是否实践? (需要实践则输入True，否则输入其他字符)')
    tmp = input()
    if tmp == 'True':
        do_input = True
    #根据是否有cuda选择模式
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if do_train:
        #创建数据集及加载
        train_dataset = BertDataset(train_text,train_label,label_2_index,max_len,tokenizer,is_test=False)
        train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)

        dev_dataset = BertDataset(dev_text, dev_label, label_2_index, max_len, tokenizer,is_test=False)
        dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
        #创建模型
        model = Bert_LSTM_NerModel(lstm_hidden,len(label_2_index)).to(device)
        opt = AdamW(model.parameters(),lr) #创建优化器

        best_score = -1
        for e in range(epoch):
            model.train()  #训练标识
            for batch_text_index,batch_label_index,batch_len in train_dataloader:
                batch_text_index = batch_text_index.to(device) #将文本索引加载到设备
                batch_label_index = batch_label_index.to(device) #将标签索引加载到设备
                #模型训练
                loss = model.forward(batch_text_index,batch_label_index)
                loss.backward()
                #优化
                opt.step()
                opt.zero_grad()

                print(f'loss:{loss:.2f}')

            #验证标识
            model.eval()
            #比对标签是否预测正确
            all_pre = []
            all_tag = []
            for batch_text_index,batch_label_index,batch_len in dev_dataloader:
                #加载入设备
                batch_text_index = batch_text_index.to(device)
                batch_label_index = batch_label_index.to(device)
                #预测
                pre = model.forward(batch_text_index)
                #引入crf则不需要进行列表转换
                # pre = pre.cpu().numpy().tolist()
                #将tag转回列表
                tag = batch_label_index.cpu().numpy().tolist()
                #提取原始标签
                for p,t,l in zip(pre,tag,batch_len):
                    p = p[1:1+l]
                    t = t[1:1+l]

                    p = [index_2_label[i] for i in p]
                    t = [index_2_label[i] for i in t]

                    all_pre.append(p)
                    all_tag.append(t)
            #计算f1_score
            f1_score = seq_f1_score(all_tag,all_pre)
            #如果f1_score是最好的，则保存该epoch训练出的模型
            if f1_score > best_score:
                torch.save(model, "best_model_crf.pt")
                best_score = f1_score
            #打印信息
            print(f"best_score:{best_score:.2f},f1_score:{f1_score:.2f}")

    if do_test:
        #加载测试集
        test_dataset = BertDataset(test_text, test_label, label_2_index, max_len, tokenizer,True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        #加载模型
        model = torch.load("best_model_crf.pt")
        #准备接收预测值、真实值
        all_pre = []
        all_tag = []
        test_out = []
        for idx,(batch_text_index, batch_label_index, batch_len) in enumerate(test_dataloader):
            text = test_text[idx]

            batch_text_index = batch_text_index.to(device)
            batch_label_index = batch_label_index.to(device)
            pre = model.forward(batch_text_index)

            # pre = pre.cpu().numpy().tolist()
            tag = batch_label_index.cpu().numpy().tolist()

            #拿到标签
            pre = pre[0][1:-1]
            tag = tag[0][1:-1]

            pre = [index_2_label[i] for i in pre]
            tag = [index_2_label[i] for i in tag]

            all_pre.append(pre)
            all_tag.append(tag)

            test_out.extend([f"{w} {t}" for w,t in zip(text,pre)])
            test_out.append("")
        #计算f1_score
        f1_score = seq_f1_score(all_tag, all_pre)
        print(f"test_f1_score:{f1_score:.2f}")
        #将文本与预测值写入文件
        with open("test_out_crf.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(test_out))

    if do_input:
        teacher_path = '../CV_cabin_origin'  #预处理后的文本
        model = torch.load("best_model_crf.pt") #加载模型
        if not os.path.exists('../CV_cabin_process'):
            os.mkdir('../CV_cabin_process')
        #逐个遍历
        for teacher_info in glob.glob(os.path.join(teacher_path, "*.txt")):
            teacher_name = os.path.basename(teacher_info).strip().split('.')[0]
            with open(teacher_info, mode='r+', encoding='utf-8') as teacher:
                url = teacher.readline() #拿url
                text = teacher.read()
                text = text[:510] #bert最多支持512字符，除去首尾填充为510
                #文本编码
                text_idx = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
                text_idx = text_idx.to(device)
                #获取预测值
                pre = model.forward(text_idx)
                #除去首尾占位
                pre = pre[0][1:-1]
                #获取原始标签
                pre = [index_2_label[i] for i in pre]
                #写入文件
                with open(f'../CV_cabin_process/{teacher_name}.txt', mode='w', encoding='utf-8') as info:
                    info.write(url)
                    info.write("\n".join([f"{w} {t}" for w,t in zip(text,pre)]))

