import parse_config as pc
import pandas as pd
import numpy as np
import math
import os

def parse_input(input_string):
    # Check if the string contains a comma
    if ',' in input_string:
        # Split the string by commas, strip whitespace, and convert to integers
        return [int(item.strip()) for item in input_string.split(',')]
    else:
        # Directly convert the string to an integer and return as a list
        return [int(input_string.strip())]

def get_flops(config_path,num_cls,start=0,end=9999):
    
    batch_size=0
    filter_size=0
    output_size_list=[]
    Memory_usage_list=[]
    input_size=[0,0,0,0] #batch, width, hight, channel
    output_size=[0,0,0,0]
    layer_flops=0
    total_flops=0
    total_memory_usage=0
   

    config_path=config_path
   

    net_defs,_,_=pc.parse_model_config(config_path)

    for id,layr in enumerate(net_defs):
        print(id,':',layr)
        if(layr['type']=='net'):
            
            batch_size=int(layr['batch'])
            
            input_size=[0,0,0,0]
            output_size=[int(layr['batch']),int(layr['width']),int(layr['height']),int(layr['channels'])]
            layer_flops=0
            Layer_memory_usage=math.prod(output_size)
            
        
        elif (layr['type']=='convolutional'):
            bn=int(layr['batch_normalize'])
            
            if(id==82 or id==94 or id==106):
                filters=int(3*(num_cls+5))
                print("Conv layr:Uisng filters: ",filters)
            else:
                filters=int(layr['filters'])
            
            filter_size=int(layr['size'])
            stride=int(layr['stride']) 
            pad=int(layr['pad'])
            
            activation=layr['activation']

            input_size=output_size_list[-1] # output of previous layer is input of current layer
            input_channels=input_size[-1] # last value of input size is channels    
            
            if filter_size==1:
                out_width=input_size[1]
                out_height=input_size[2]
            else:
                
                
                out_width=int((input_size[1]-filter_size+2*pad)/stride)+1
                out_height=int((input_size[2]-filter_size+2*pad)/stride)+1
            
            out_channels=filters
            
            output_size=[batch_size,out_width,out_height,out_channels]

            convolution_flops=(2*(filter_size*filter_size*input_channels*out_height*out_width)*out_channels)
            
            if(bn==1):
                mean_cal_flops=input_channels*(batch_size*input_size[1]*input_size[2]-1)
                var_cal_flops=input_channels*2*(batch_size*input_size[1]*input_size[2])
                norm_cal_flops=input_channels*2*(batch_size*input_size[1]*input_size[2])
                scale_n_shif_cal_flops=input_channels*2*(batch_size*input_size[1]*input_size[2])
                
                total_bn_flops=mean_cal_flops+var_cal_flops+norm_cal_flops+scale_n_shif_cal_flops
            else:
                total_bn_flops=0
            #print("total_bn_flops: ",total_bn_flops)

            activation_flops=3*input_size[0]*input_size[1]*input_size[2]*input_size[3]

            input_size=output_size_list[-1]
            output_size=[batch_size,out_width,out_height,out_channels]
            layer_flops=(convolution_flops+total_bn_flops+activation_flops)
            Layer_memory_usage=math.prod(output_size)
                
               
        elif (layr['type']=='shortcut'):
             input_string=layr['from'] # a list of layer id 
             layers=parse_input(input_string)
            
             input_size=output_size_list[layers[0]] # short cut is to take the input of mentioned layer(Lm) thus output of previous layr
             output_size=output_size_list[layers[0]] ## of Lm ie Lm-1
             layer_flops=0
             Layer_memory_usage=math.prod(output_size)

        elif (layr['type']=='route'):
            #print("routing layer sizes")
            input_string=layr['layers'] # a list of layer id
            layers=parse_input(input_string)

            for layer_idx in layers:
                if layer_idx>0:
                    layer_idx=layer_idx-1
                #print(output_size_list[layer_idx])
                input_size=output_size_list[layer_idx]
                output_size=output_size_list[layer_idx]
            
            input_size=output_size_list[layer_idx]
            output_size=output_size_list[layer_idx]
            layer_flops=0
            Layer_memory_usage=math.prod(output_size)


        elif (layr['type']=='upsample'):
            stride=int(layr['stride'])
            mask=[1,stride,stride,1]
            
            input_size=output_size_list[-1]
            output_size=[int(m*i)for m,i in zip(mask,input_size)]
            layer_flops=0
            Layer_memory_usage=math.prod(output_size)

        elif (layr['type']=='yolo'):
            
            W=output_size_list[-1][1]# Feature width
            H=output_size_list[-1][2]# Feature Height
            A=3 #number of anchors
            
            classes=num_cls
            print("yolo layr:Uisng num classes: ",classes)
            Predictions_per_anchor=5+classes#int(layr['classes'])
            
            Total_predictions=batch_size*W*H*A*Predictions_per_anchor
            FLOPS_sigmoid=4*Total_predictions
            FLOPS_bbox_transform=batch_size*W*H*A*4
            

            input_size=output_size_list[-1]
            output_size=[batch_size,H,W,int(A*Predictions_per_anchor)]
            layer_flops=FLOPS_sigmoid+FLOPS_bbox_transform
            Layer_memory_usage=math.prod(output_size)

        
        output_size_list.append(output_size)
        
        Memory_usage_list.append(Layer_memory_usage)
        
        if id>=start and id <=end:
            total_flops+=layer_flops
            total_memory_usage+=Layer_memory_usage

            
        
        print("input_size: ","    output_size: ")
        print(input_size,output_size)
        print("layer_flops: ",layer_flops/10**12 ,"TFLOPS")
    print("total_flops: ",total_flops/10**9 ,"GFLOPS","Memory usage: ",total_memory_usage/10**6)

    return [(total_flops/10**9),(total_memory_usage/10**6)]



if __name__=='__main__':

  
    

    file_name='yolov3.cfg'
    config_path=os.path.join(os.getcwd(),'config',file_name)
    print("config_path",config_path)
    output_list=[]
    num_class=8
    for i in range (1,num_class+1):
    
        (f,m)=get_flops(config_path,i)
        output_list.append((f-54.32,m-761.45)) # darknet flops and memroy removed
    for ouput in output_list:
        print(ouput[0],",",ouput[1])
    
    pdf=pd.DataFrame(output_list,columns=["BFLOPS","MB"])
    file_name='flops_n_memory.csv'
    pdf.to_csv(file_name,index=False)
    print(output_list)