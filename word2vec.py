import collections
import math
import os
import random
import zipfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import urllib
import tensorflow as tf

# 网上下载数据
url = 'http://mattmahoney.net/dc/'

# 下载数据
def maybe_download(filename, excepted_bytes):
    # 判断文件是否存在，若不存在则下载
    if not os.path.exists(filename):
        # 数据集不存在，开始下载
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    # 核对文件尺寸
    statinfo = os.stat(filename)
    if statinfo.st_size == excepted_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify' + filename + '.Can you get to it with a browser?'
        )
    return filename

# 测试文件是否存在
filename = maybe_download('text8.zip', 31344016)

# 解压下载的压缩文件，并使用tf.compat.as_str将数据转成单词的列表
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
words = read_data(filename)
print('Data size:', len(words))

# 创建词汇表
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    # collections.Counter()返回的是形如[["unkown",-1],("the",4),("physics",2)
    # 使用collections.Counter统计单词列表中单词的频数
    # 然后使用most_common方法取top 50000频数的单词作为vocabulary
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    # 将top 50000词汇的vocabulary放入dictionary中，以便快速查询
    # 将全部单词转为编号（以频数排序的编号）
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    # top 50000词汇以外的单词，我们认定其为unkown，将其编号为0，并统计这类词汇的量
    unk_count = 0
    for word in words:# 遍历单词列表
        # 对其中每一个单词，先判断是否出现在dictionary中
        if word in dictionary:
            # 如果出现在字典中，则转换成它的编号
            index = dictionary[word]
        else:
            # 如果不在字典中，则转为编号0
            index = 0
            unk_count += 1
        data.append(index)
    # 将统计好的unknown的单词数，填入count
    count[0][1] = unk_count
    # 将字典进行翻转,形如：{3:"the,4:"an"}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionary = build_dataset(words)

# 删除原始单词列表，节约内存
del words
# 打印vocabulary中最高频出现的词汇及其数量
print('most common words (+UNK)', count[:5])
# 将已经转换为编号的数据进行输出，从data中输出频数，从翻转字典中输出编号对应的单词
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# 使用Skip-Gram生成word2vec的训练样本
data_index = 0
# 定义函数generate_batch用来生成训练用的batch数据
# 其中参数batch_size为batch的大小，skip-window为单词最远可以联系的距离
# num_skips为对每个单词生成多少个样本
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # 设置num_skips的满足条件
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # 用np.ndarray将batch和labels初始化为数组
    batch = np.ndarray(shape=(batch_size), dtype = np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # span为对某个单词创建相关样本时会使用到的单词数量，包括目标单词本身和它前后的词
    span = 2 * skip_window + 1
    # 创建双向队列，最大长度为span
    buffer = collections.deque(maxlen=span)
    # 在双向队列中填入初始值
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1)% len(data)
    # 进入第一次循环，i表示第几次入双向队列
    for i in range(batch_size // num_skips):
        target = skip_window # 定义buffer中第skip_window个单词是目标单词
        # 定义生成样本时需要避免的单词列表，因为我们要预测的是语境单词，不包括目标单词本身
        targets_to_avoid = [skip_window]
        # 进入第二次循环，每次循环中对一个语境单词产生样本，先产生随机数，直到不在需要避免的单词中，也即需要找到可以使用的语境词语
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span-1)
            targets_to_avoid.append(target)# 因为该语境单词已经使用过了，所以再把它添加到targets_to_void中过滤
            batch[i * num_skips + j] = buffer[skip_window] # 目标词汇
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index+1)%len(data)
    return batch, labels
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range (8):
    print("目标单词："+reverse_dictionary[batch[i]]+"对应编号为："+str(batch[i])+
          "对应的语境单词为: "+reverse_dictionary[labels[i,0]]+"编号为："+str(labels[i,0]))

# 定义训练时的参数
batch_size = 128
embedding_size = 128  # 将单词转为稠密向量的维度
skip_window = 1
num_skips = 2

# 定义验证数据的一些参数
valid_size = 16  # 用来抽取的验证单词数
valid_window = 100  # 指验证单词只从频数最高的100个单词中抽取
valid_examples = np.random.choice(valid_window, valid_size, replace=False)  # 进行随机抽取
num_sampled = 64  # 指训练时用来做负样本的噪声单词的数量

# 开始定义Skip-Gram Word2Vec模型的网络结构
# 创建一个tf.Graph并设置为默认的graph，作为计算图。
# 同时为输入数据和标签申请占位符，并将验证样例的随机数保存成TensorFlow的常数
graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    # 限定运算在cpu上运行
    with tf.device('/cpu:0'):
        # 单词大小为50000，向量维度为128，随机采样在（-1，1）之间的浮点数
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # 使用tf.nn.embedding_lookup()函数查找train_inputs对应的向量embed
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # 优化目标选择NCE Loss
        # 使用tf.truncated_normal初始化NCE Loss中的权重参数nce_weights，并将其初始化为0
        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # 计算学习出的词向量embedings在训练数据上的loss，并使用tf.reduce_mean进行汇总
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    # 定义优先器为SGD，且学习速率为1.0，然后计算嵌入向量embeddings的L2范数norm，并计算出标准化后的normalized_embeddings
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    # 查询验证单词的嵌入向量
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # 计算相似性
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
    # 初始化所有模型参数
    init = tf.global_variables_initializer()

# 启动训练
num_steps = 10000  # 迭代次数为10万次
# 创建并设置默认的session
with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    # 开始迭代训练
    average_loss = 0
    for step in range(num_steps):
        # 生成一个batch的Inputs和labels数据，并用它们创建feed_dict
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        # 使用session执行优化器运算和损失计算
        _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += loss_val

        # 每2000次循环，计算一下平均loss并显示出来
        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("第{}轮迭代后的损失为：{}".format(step, average_loss))
        # 每10000次循环，计算一次验证单词与全部单词的相似度，并将与每个验证单词最相似的8个单词展示出来
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "与单词 {} 最相似的： ".format(str(valid_word))
                for k in range(top_k):
                    closed_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," %(log_str, closed_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

# Word2Vec的可视化
# low_dim_embs是降维到2维的单词的空间向量
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "标签数超过了嵌入向量的个数"
    plt.figure(figsize=(16, 16))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x,y)  # 显示散点图
        # 显示单词本身
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom'
                     )
    plt.savefig(filename)

# 降维
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)
plt.show()