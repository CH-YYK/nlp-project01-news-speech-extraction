# -*- coding: utf-8 -*-
import os
LTP_DATA_DIR = '/home/yikang/Desktop/TextData/Project01-NLP-LTP/nlp-project01/ltp_data_v3.4.0'  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
import jieba
from pyltp import Postagger
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import SementicRoleLabeller
from pyltp import Segmentor

postagger = Postagger() # 初始化实例

def get_parsing(sentence):
    postagger.load(pos_model_path)  # 加载模型
    words=list(pyltp_cut(sentence)) #结巴分词
    postags = list(postagger.postag(words))  # 词性标注

    tmp=[str(k+1)+'-'+v for k,v in enumerate(words)]
    print('\t'.join(tmp))
    parser = Parser() # 初始化实例
    parser.load(par_model_path)  # 加载模型
    arcs = parser.parse(words, postags)  # 句法分析
    # for arc in arcs:
    #     if arc.relation=='SBV':
    #         name=

    print ("\t".join("%s-%d:%s" % (k+1,arc.head, arc.relation) for k,arc in enumerate(arcs)))
    parser.release()  # 释放模型
    return arcs

def get_name_entity(sentence):
    recognizer = NamedEntityRecognizer()  # 初始化实例
    recognizer.load(ner_model_path)  # 加载模型
    words = list(pyltp_cut(sentence))  # 结巴分词
    postags = list(postagger.postag(words))  # 词性标注
    netags = recognizer.recognize(words, postags)  # 命名实体识别
    tmp=[str(k+1)+'-'+v for k,v in enumerate(netags)]
    print('\t'.join(tmp))
    recognizer.release()  # 释放模型

def get_srl(sentence):
    labeller = SementicRoleLabeller()  # 初始化实例
    labeller.load(srl_model_path)  # 加载模型
    words = list(pyltp_cut(sentence))  # pyltp分词
    postags = list(postagger.postag(words))  # 词性标注
    arcs=get_parsing(sentence)
    # arcs 使用依存句法分析的结果
    roles = labeller.label(words, postags, arcs)  # 语义角色标注

    # 打印结果
    for role in roles:
        print(role.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments]))
        labeller.release()  # 释放模型
#pyltp中文分词
def pyltp_cut(sentence):

    segmentor = Segmentor()  # 初始化实例
    segmentor.load(cws_model_path)  # 加载模型
    words = segmentor.segment(sentence)  # 分
    segmentor.release()  # 释放模型
    return words

# text=['澳大利亚战略政策研究所负责人彼得·詹宁斯也指责称，基廷正在经历“特朗普时刻”——特朗普经常与美国的情报部门主管发生冲突','萨瑟斯表示所有的狗都受过专门的训练及经过严格评估','林瑞希认为，“平时我们老师上课不拘泥于课本，注重拓展，引导我们思考生物学科的前沿问题，而这正好符合牛津、剑桥的招生要求','一位同学对记者坦言大学期间从未拉过女孩子的手']
# text=['《环球报》称，该法令最初的目的是放松对收藏家与猎人的限制，但现在扩大到其他条款。新法令将普通公民购买枪支的弹药数量上限提高至每年5000发，此前这一上限是每年50发。博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”。另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”']
text=["据巴西《环球报》7日报道，巴西总统博索纳罗当天签署行政法令，放宽枪支进口限制，并增加民众可购买弹药的数量。\r\n《环球报》称，该法令最初的目的是放松对收藏家与猎人的限制，但现在扩大到其他条款。新法令将普通公民购买枪支的弹药数量上限提高至每年5000发，此前这一上限是每年50发。博索纳罗在法令签署仪式上称，“我们打破了垄断”“你们以前不能进口，但现在这些都结束了”。另据法新社报道，当天在首都巴西利亚的一次集会上，博索纳罗还表示，“我一直说，公共安全从家里开始的。”\r\n这不是巴西第一次放宽枪支限制。今年1月，博索纳罗上台后第15天就签署了放宽公民持枪的法令。根据该法令，希望拥有枪支的公民须向联邦警察提交申请，通过审核者可以在其住宅内装备最多4把枪支，枪支登记有效期由5年延长到10年。《环球报》称，博索纳罗在1月的电视讲话中称，要让“好人”更容易持有枪支。“人民希望购买武器和弹药，现在我们不能对人民想要的东西说不”。\r\n2004年，巴西政府曾颁布禁枪法令，但由于多数民众反对，禁令被次年的全民公投否决。博索纳罗在参加总统竞选时就表示，要进一步放开枪支持有和携带条件。他认为，放宽枪支管制，目的是为了“威慑猖狂的犯罪行为”。资料显示，2017年，巴西发生约6.4万起谋杀案，几乎每10万居民中就有31人被杀。是全球除战争地区外最危险的国家之一。\r\n不过，“以枪制暴”的政策引发不少争议。巴西《圣保罗页报》称，根据巴西民调机构Datafolha此前发布的一项调查，61%的受访者认为应该禁止持有枪支。巴西应用经济研究所研究员塞奎拉称，枪支供应增加1%，将使谋杀率提高2%。1月底，巴西民众集体向圣保罗联邦法院提出诉讼，质疑博索纳罗签署的放宽枪支管制法令。\r\n巴西新闻网站“Exame”称，博索纳罗7日签署的法案同样受到不少批评。公共安全专家萨博称，新的法令扩大了少数人的特权，不利于保护整个社会。（向南）\r\n"]

for t in text:
    get_parsing(t)
    get_name_entity(t)
    words = list(pyltp_cut(t))  # pyltp分词
    postags = list(postagger.postag(words))  # 词性标注
    pos=[str(k+1)+'-'+str(v) for k,v in enumerate(postags)]
    print(pos)



