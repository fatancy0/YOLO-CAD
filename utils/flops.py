import torch
from nets.yolo4_tiny import  YoloBody,YoloBody2,YoloBody3,YoloBody5
from utils.FLOPcount  import   get_model_complexity_info


def main(  input_shape=(3, 416, 416)):
    model =  YoloBody5(6,20)
    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops*2}\nParams: {params}\n{split_line}')


if __name__ == '__main__':

    main(
         input_shape=(3, 416, 416)
         )
