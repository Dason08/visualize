import pandas as pd
import numpy as np
import cv2
import argparse
import keras
import time
import sys
from keras import backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.applications.resnet50 import ResNet50

K.set_learning_phase(1) #set learning phase

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path',type=str,help='Input image path')
    parser.add_argument('--size', type=int, default=224,help='Image size (default = 224×224)/ InceptionV3とXception使用時は299推奨')
    parser.add_argument('--model',type=str,default='ResNet50',
    help='使用可能なモデルは以下の10種類 : ResNet50(default) / VGG16 / VGG19 / DenseNet121[使用不可] / Xception /  InceptionV3 / InceptionResNetV2[非推奨] / MobileNet[非推奨] / MobileNetV2[非推奨]/ NASNet[使用不可]')

    args = parser.parse_args()

    return args

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer = layer_dict[layer_name]
    return layer


def Grad_Cam(input_model, x, layer_name, target_size):
    '''
    Args:
       input_model: モデルオブジェクト
       x: 画像(array)
       layer_name: 畳み込み層の名前

    Returns:
       jetcam: 影響の大きい箇所を色付けした画像(array)

    '''

    model = input_model

    # 前処理
    X = np.expand_dims(x, axis=0)

    X = X.astype('float32')
    preprocessed_input = X / 255.0


    # 予測クラスの算出

    predictions = model.predict(preprocessed_input)
    class_idx = np.argmax(predictions[0])
    class_output = model.layers[-1].output  #costに相当する？

    #  勾配を取得
    conv_output = model.get_layer(layer_name).output   # layer_nameのレイヤーのアウトプット
    grads = K.gradients(class_output, conv_output)[0]  # gradients(loss, variables) で、variablesのlossに関しての勾配を返す
    #first_derivative：１階微分
    first_derivative = K.exp(class_output)[0][class_idx] * grads
    #second_derivative：２階微分
    second_derivative = K.exp(class_output)[0][class_idx] * grads * grads
    #third_derivative：３階微分
    third_derivative = K.exp(class_output)[0][class_idx] * grads * grads * grads

    gradient_function = K.function([model.input], [conv_output, first_derivative, second_derivative, third_derivative])  # model.inputを入力すると、conv_outputとgradsを出力する関数


    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = gradient_function([preprocessed_input])
    conv_output, conv_first_grad, conv_second_grad, conv_third_grad = conv_output[0], conv_first_grad[0], conv_second_grad[0], conv_third_grad[0]

    global_sum = np.sum(conv_output.reshape((-1, conv_first_grad.shape[2])), axis=0)

    alpha_num = conv_second_grad
    alpha_denom = conv_second_grad*2.0 + conv_third_grad*global_sum.reshape((1,1,conv_first_grad.shape[2]))
    alpha_denom = np.where(alpha_denom!=0.0, alpha_denom, np.ones(alpha_denom.shape))
    alphas = alpha_num / alpha_denom


    weights = np.maximum(conv_first_grad, 0.0)
    alphas_thresholding = np.where(weights, alphas, 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis = 0), axis = 0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))

    alphas /= alpha_normalization_constant_processed.reshape((1,1,conv_first_grad.shape[2]))

    deep_linearization_weights = np.sum((weights * alphas).reshape((-1, conv_first_grad.shape[2])))

    grad_CAM_map = np.sum(deep_linearization_weights * conv_output, axis=2)

    cam = np.maximum(grad_CAM_map, 0)
    cam = cam / np.max(cam)

    #cam = resize(cam, (224,224))

    cam = cv2.resize(cam, (target_size, target_size), cv2.INTER_LINEAR) # 画像サイズは200で処理したので

    jetcam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)  # モノクロ画像に疑似的に色をつける
    #jetcam = cv2.cvtColor(jetcam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
    jetcam = (np.float32(jetcam) + x / 2)   # もとの画像に合成

    return jetcam


def set_models(modelname):
    if modelname == "ResNet50":
        model = ResNet50(weights = 'imagenet')
        target_layer = 'activation_49'
    elif modelname == "VGG16":
        model = VGG16(weights = 'imagenet')
        target_layer = 'block5_conv3'
    elif modelname == "VGG19":
        model = VGG19(weights = 'imagenet')
        target_layer = 'block5_conv4'
    elif modelname == 'DenseNet121':
        model = DenseNet121(weights = 'imagenet')#やばいよやばいよやばいよやばいよやばいよ
        target_layer = 'bn'
    elif modelname == 'Xception':
        model = Xception(weights = 'imagenet')
        target_layer = 'block14_sepconv2'
    elif modelname == 'InceptionV3':
        model = InceptionV3(weights = 'imagenet')
        target_layer = 'mixed10'
    elif modelname == "MobileNet":
        model = MobileNet(input_weights = 'imagenet')
        target_layer = 'conv_pw_13'#やばいよやばいよやばいよやばいよやばいよ
    elif modelname == "MobileNetV2":
        model = MobileNetV2(weights = 'imagenet')
        target_layer = 'Conv_1'#やばいよやばいよやばいよやばいよやばいよ
    elif modelname == "NASNet":
        model = NASNetMobile(weights = 'imagenet')
        target_layer = 'activation_188'#やばいよやばいよやばいよやばいよやばいよ
    elif modelname == "InceptionResNetV2":
        model = InceptionResNetV2(weights = 'imagenet')#やばいよやばいよやばいよやばいよやばいよ
        target_layer = 'conv_7b_ac'
    else:
        print("Model Error：あなたの選択した%sは使用できません。"%modelname)
        sys.exit()

    return model,target_layer


if __name__ == '__main__':
    args = get_args()
    model, target_layer = set_models(args.model)
    img = img_to_array(load_img(args.image_path,target_size=(args.size,args.size)))
    img_GCAM = Grad_Cam(model, img, target_layer, args.size)
    time = time.ctime()
    img_Gname = args.image_path+time+"_GCAM++_%s.jpg"%args.model
    cv2.imwrite(img_Gname, img_GCAM)
    print("Completed.")
