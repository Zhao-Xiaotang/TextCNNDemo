"""
    完成对数据的处理，包括去除标点符号等
"""
import string
import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models.keyedvectors import KeyedVectors

def replacePunct(String):
    """
    利用String的标点符号库去除标点符号
    """
    punctuation = string.punctuation
    for i in punctuation:
        String = String.replace(i, '')

    return String

def string2list(df_data):
    """
    paras:
    input:
    data_json: the list of sample jsons

    outputs:
    data_text: the list of word list
    data_label: label list
    返回去除了标点符号的评论列表，以及标签列表
    """
    data_text = [list(replacePunct(text)) for text in df_data['Notes']]
    data_label = [int(text) for text in df_data['Label ID']]
    return data_text, data_label

# 读取数据集，验证数量
df_label_data = pd.read_csv('./data/Detection/labels.csv')
df_train_data = pd.read_csv('./data/Detection/train.csv')
df_dev_data = pd.read_csv('./data/Detection/dev.csv')
train_text, train_label = string2list(df_train_data)
dev_text, dev_label = string2list(df_dev_data)


df_test_data = pd.read_csv('./data/Detection/test.csv')
test_text = [list(replacePunct(text)) for text in df_test_data['Notes']]



# 定义tokenizer并使用准备好的文本序列进行拟合
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=None,
    filters=' ',
    lower=True,
    split=' ',
    char_level=False,
    oov_token='UNKNOW',
    document_count=0
)
tokenizer.fit_on_texts(train_text)

"""
    定义batch_size, 序列最大长度
    将字符串序列转为整数序列
    将序列按照最大长度填充
    准备label　tensor
    准备 train_dataset, dev_dataset
"""
BATCH_SIZE = 64
MAX_LEN = 500
BUFFER_SIZE = tf.constant(len(train_text), dtype=tf.int64)

# text 2 lists of int
train_sequence = tokenizer.texts_to_sequences(train_text)
dev_sequence = tokenizer.texts_to_sequences(dev_text)

# padding sequence
train_sequence_padded = pad_sequences(train_sequence, padding='post', maxlen=MAX_LEN)
dev_sequence_padded = pad_sequences(dev_sequence, padding='post', maxlen=MAX_LEN)

# cvt the label tensors
train_label_tensor = tf.convert_to_tensor(train_label, dtype=tf.float32)
dev_label_tensor = tf.convert_to_tensor(dev_label, dtype=tf.float32)

# create the dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_sequence_padded, train_label_tensor)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(BUFFER_SIZE)
dev_dataset = tf.data.Dataset.from_tensor_slices((dev_sequence_padded, dev_label_tensor)).batch(BATCH_SIZE, drop_remainder=True).prefetch(BUFFER_SIZE)

"""
    构建TextCNN模型
"""

# 构建模型
VOCAB_SIZE = len(tokenizer.index_word) + 1   # 词典大小
EMBEDDING_DIM = 300                          # 词向量大小
FILTERS = [3, 4, 5]                          # 卷积核尺寸个数
FILTER_NUM = 256                             # 卷积层卷积核个数
CLASS_NUM = 2                                # 类别数
DROPOUT_RATE = 0.8                           # dropout比例

# get_embeddings: 读取预训练词向量
# PretrainedEmbedding: 构建加载预训练词向量且可fine tuneEmbedding Layer

def get_embeddings():
    pretrained_vec_path = "./saved_model/wiki-news-300d-1M-subword.vec"
    word_vectors = KeyedVectors.load_word2vec_format(pretrained_vec_path, binary=False)
    word_vocab = set(word_vectors.vocab.keys())
    embeddings = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)
    for i in range(len(tokenizer.index_word)):
        i += 1
        word = tokenizer.index_word[i]
        if word in word_vocab:
            embeddings[i, :] = word_vectors.get_vector(word)
    return embeddings

class PretrainedEmbedding(tf.keras.layers.Layer):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, embeddings, rate=0.1):
        super(PretrainedEmbedding, self).__init__()
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.embeddings_initializer = tf.constant_initializer(embeddings)
        self.dropout = tf.keras.layers.Dropout(rate)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.VOCAB_SIZE, self.EMBEDDING_DIM),
            initializer=self.embeddings_initializer,
            dtype=tf.float32
        )

    def call(self, x, trainable=None):
        output = tf.nn.embedding_lookup(
            params=self.embeddings,
            ids=x
        )
        return self.dropout(output, training=trainable)

embeddings = get_embeddings()


# 模型构建
class TextCNN(tf.keras.Model):
    def __init__(self, VOCAB_SIZE, EMBEDDING_DIM, FILTERS, FILTER_NUM, CLASS_NUM, DROPOUT_RATE, embeddings):
        super(TextCNN, self).__init__()
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.FILTERS = FILTERS
        self.FILTER_NUM = FILTER_NUM
        self.CLASS_NUM = CLASS_NUM
        self.DROPOUT_RATE = DROPOUT_RATE

        self.embed = PretrainedEmbedding(self.VOCAB_SIZE, self.EMBEDDING_DIM, embeddings)
        self.convs = []
        self.max_pools = []
        for i, FILTER in enumerate(self.FILTERS):
            conv = tf.keras.layers.Conv1D(self.FILTER_NUM, FILTER,
                                          padding='same', activation='relu', use_bias=True)
            max_pool = tf.keras.layers.GlobalAveragePooling1D()
            self.convs.append(conv)
            self.max_pools.append(max_pool)
        self.dropout = tf.keras.layers.Dropout(self.DROPOUT_RATE)
        self.fc = tf.keras.layers.Dense(self.CLASS_NUM, activation='softmax')

    def call(self, x):
        x = self.embed(x, trainable=True)
        conv_results = []
        for conv, max_pool in zip(self.convs, self.max_pools):
            conv_results.append(max_pool(conv(x)))
        x = tf.concat(conv_results, axis=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

textcnn = TextCNN(VOCAB_SIZE, EMBEDDING_DIM, FILTERS, FILTER_NUM, CLASS_NUM, DROPOUT_RATE, embeddings)

# 定义损失函数 优化器
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.0005)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
eval_loss = tf.keras.metrics.Mean(name='eval_loss')
eval_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='eval_accuracy')

# 定义单步训练、测试函数
@tf.function
def train_step(input_tensor, label_tensor):
    with tf.GradientTape() as tape:
        prediction = textcnn(input_tensor)
        loss = loss_object(label_tensor, prediction)
    gradients = tape.gradient(loss, textcnn.trainable_variables)
    optimizer.apply_gradients(zip(gradients, textcnn.trainable_variables))

    train_loss(loss)
    train_accuracy(label_tensor, prediction)


@tf.function
def eval_step(input_tensor, label_tensor):
    prediction = textcnn(input_tensor)
    loss = loss_object(label_tensor, prediction)

    eval_loss(loss)
    eval_accuracy(label_tensor, prediction)

# 定义writer
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time + '/train'
test_log_dir = 'logs/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


# tokenizer.fit_on_texts(test_text)
test_sequence = tokenizer.texts_to_sequences(test_text)
# padding sequence
test_sequence_padded = pad_sequences(test_sequence, padding='post', maxlen=MAX_LEN)
prediction = textcnn.predict(test_sequence_padded)


# 模型训练、保存权重
EPOCHS = 10
for epoch in range(EPOCHS):

    train_loss.reset_states()
    train_accuracy.reset_states()
    eval_loss.reset_states()
    eval_accuracy.reset_states()

    for batch_idx, (train_input, train_label) in enumerate(train_dataset):
        train_step(train_input, train_label)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for batch_idx, (dev_input, dev_label) in enumerate(dev_dataset):
        eval_step(dev_input, dev_label)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', eval_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', eval_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {:.4f}, Accuracy: {:.4f}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'
    print(template.format(epoch + 1,
                          train_loss.result().numpy(),
                          train_accuracy.result().numpy() * 100,
                          eval_loss.result().numpy(),
                          eval_accuracy.result().numpy() * 100))
    textcnn.save_weights('./saved_model/weights_{}.h5'.format(epoch))
    textcnn.save('./model/textcnn_{}.h5'.format(epoch))
