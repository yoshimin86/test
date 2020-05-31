import argparse 

import tensorflow as tf 
import torch 

from i3dpt import I3D
from i3dtf import InceptionI3d

def transfer_weights(tf_checkpoint,pt_checkpoint, batch_size, frame_nb, class_nb, modality='rgb'):
    im_size = 224
    in_channels = 3    
    i3d_pt = I3D(class_nb,modality=modality)

    if modality =='rgb':
        scope = 'RGB'
    elif modality == 'flow':
        scope = 'Flow'
    
    with tf.variable_scope(scope):
        rgb_model = InceptionI3d(class_nb,final_endpoint='Logits')
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(batch_size,frame_nb,im_size,im_size,in_channels)
        )
        rgb_logits,_ = rgb_model(rgb_input,is_training=False,dropout_keep_prob=1.0)

    rgb_variable_map = {}
    for variable in tf.global_variables():
        if variable.name.split('/')[0] == scope:
            rgb_variable_map[variable.name.replace(':0','')] = variable
    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    
    with tf.Session() as sess:
        rgb_saver.restore(sess,tf_checkpoint)

        i3d_pt.eval()
        i3d_pt.load_tf_weights(sess)
        i3d_pt_state_dict = i3d_pt.cpu().state_dict()
        torch.save(i3d_pt_state_dict,pt_checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Transfers the kinetics rgb pretrained i3d\
    inception v1 weights from tensorflow to pytorch and saves the weights as\
    as state_dict')
    parser.add_argument(
        '--rgb', action='store_true', help='Convert RGB pretrained network')
    parser.add_argument(
        '--rgb_tf_checkpoint',
        type=str,
        default='nAction_TF_I3D_maker/weights/my_RGB_model.ckpt',
        help='Path to tensorflow weight checkpoint trained on rgb')
    parser.add_argument(
        '--rgb_pt_checkpoint',
        type=str,
        default='nAction_TF_I3D_maker/weights/my_RGB_model.pth',
        help='Path for pytorch state_dict saving')
    parser.add_argument(
        '--flow', action='store_true', help='Convert Flow pretrained network')
    parser.add_argument(
        '--flow_tf_checkpoint',
        type=str,
        default='model/tf_flow_imagenet/model.ckpt',
        help='Path to tensorflow weight checkpoint trained on flow')
    parser.add_argument(
        '--flow_pt_checkpoint',
        type=str,
        default='model/model_flow.pth',
        help='Path for pytorch state_dict saving')
    parser.add_argument(
        '--batch_size',
        type=int,
        default='2',
        help='Batch size for comparison between tensorflow and pytorch outputs')
    parser.add_argument(
        '--frame_nb',
        type=int,
        default='16',
        help='Batch size for comparison between tensorflow and pytorch outputs')
    parser.add_argument(
        '--class_nb',
        type=int,
        default='5',
        help='Batch size for comparison between tensorflow and pytorch outputs')
    
    args = parser.parse_args()

    if args.rgb:
        transfer_weights(
            args.rgb_tf_checkpoint,
            args.rgb_pt_checkpoint,
            batch_size=args.batch_size,
            frame_nb = args.frame_nb,
            class_nb = args.class_nb,
            modality='rgb')
    if args.flow:
        transfer_weights(
            args.flow_tf_checkpoint,
            args.flow_pt_checkpoint,
            batch_size=args.batch_size,
            frame_nb = args.frame_nb,
            class_nb = args.class_nb,
            modality='flow')