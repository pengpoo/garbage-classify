# -*- coding: utf-8 -*-

from tools.cust_utils import data_flow
from absl import flags, app
from models.resnet50 import ResNet50
from keras.optimizer_v2.adam import Adam
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras import regularizers

flags.DEFINE_string('data_dir', './garbage_classify_v2/train_data_v2', '图片文件夹路径')
flags.DEFINE_integer('num_classes', 40, '分类任务类别数')
flags.DEFINE_integer('input_size', 224, '模型输入图片大小')
flags.DEFINE_integer('batch_size', 32, '批量大小')
flags.DEFINE_float('learning_rate', 1e-3, '学习率大小')
flags.DEFINE_integer('epochs', 8, 'epochs 大小')
flags.DEFINE_string('output_dir', './output/', '训练后模型保存的文件夹')
FLAGS = flags.FLAGS


def add_custom_top_layers():
    # 加载预训练模型，设置 include_top=False 排除网络的顶部，
    # 包括池化层（如果 pooling 有设置值）和全链接输出层
    base_model = ResNet50(weights='imagenet',
                          include_top=False,
                          pooling=None,
                          input_shape=(224, 224, 3),
                          classes=40)
    # 在开始训练前冻结预训练层的权重
    for layer in base_model.layers:
        layer.trainable = False
    # 添加自己的网络层
    # 可以直接一个 Dense 加 softmax，也不知道多几个全链接层收益有多少
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pooling')(x)  # -> (2048,)
    x = Dropout(0.5, name='dropout1')(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(40, activation='softmax')(x)
    # 编译模型 优化器选择的是 Adam
    opt = Adam(learning_rate=0.001, decay=0.0005)
    model = Model(base_model.input, x)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(_argv):
    train_sequence, validation_sequence = data_flow(data_dir=FLAGS.data_dir,
                                                    batch_size=FLAGS.batch_size,
                                                    num_classes=FLAGS.num_classes,
                                                    input_size=FLAGS.input_size)
    model = add_custom_top_layers()
    # 开始训练
    history = model.fit_generator(
        train_sequence,
        steps_per_epoch=len(train_sequence),
        epochs = FLAGS.epochs,
        validation_data=validation_sequence,
        validation_steps=len(validation_sequence),
        callbacks=[
            ModelCheckpoint('./output/best.h5', monitor='val_loss', save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, mode='min'),
            EarlyStopping(monitor='val_loss', patience=10)
        ]
    )
    # 这里在 8 个 epoch 后，模型的验证精度达到了 80% 左右，并取得的进步在变小，
    # 添加的顶层网络现在已经经过了训练，现在可以选择解冻所有层（或者靠近输出的部分层），
    # 再以一个较小的学习率来继续训练，可能需要较长的时间
    print('=======done training=======')


if __name__ == "__main__":
    app.run(train_model)
    # 可以用下面两行看网络结构，注释掉上面一行（已经训练过且不想重新训练），Output Shape 里 None 所在的维度是 batch_dim
    # model = add_custom_top_layers()
    # model.summary()
