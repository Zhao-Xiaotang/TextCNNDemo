"""
    根据建立好的模型，进行预测
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd
import string


print('加载模型')
model = load_model('./saved_model/weights_4.h5')
optimizer = tf.keras.optimizers.Adam(0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# 测试test集

df_label_data = pd.read_csv('./data/Detection/labels.csv')
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
    return data_text

# 读取数据集，验证数量
df_test_data = pd.read_csv('./data/Detection/test.csv')
test_text = string2list(df_test_data)

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
tokenizer.fit_on_texts(test_text)

"""
    定义batch_size, 序列最大长度
    将字符串序列转为整数序列
    将序列按照最大长度填充
    准备label　tensor
    准备 train_dataset, dev_dataset
"""
BATCH_SIZE = 64
MAX_LEN = 500
BUFFER_SIZE = tf.constant(len(test_text), dtype=tf.int64)

# text 2 lists of int
test_sequence = tokenizer.texts_to_sequences(test_text)

# padding sequence
test_sequence_padded = pad_sequences(test_sequence, padding='post', maxlen=MAX_LEN)


prediction = model.predict(test_sequence_padded)
print(len(prediction))