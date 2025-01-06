# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:48:04 2019

@author: Keshik

Sources 
    https://github.com/eriklindernoren/PyTorch-YOLOv3
"""

def parse_model_config(path):
    """
        Parses the yolo-v3 layer configuration file and returns module definitions
        
        Args
            path: configuration file path
            
        Returns
            Module definition as a list
    
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    #lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x for x in lines if x ]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    #print(lines)
    darkNet_module_defs = []
    YOLONet_module_defs=[]
    net_defs=[] # complete net ie hyper paerameter + darknet _defs+YOLO_defs 
    module_defs=darkNet_module_defs
    
    c,s,r,y,u,o=(0,0,0,0,0,0) #cont for conv,shortcut,route,upsample,others
    
    print('parsing cfg file...')
    for line in lines:
        #print(line)
        
        if line.startswith('#'):
            # '##############' line or # comment line if line ends with # then it is darknet - yolonet separatio
            # line else it is a commne
            if line.endswith('#'):
                module_defs=YOLONet_module_defs
            else:
                continue
        
        elif line.startswith('['): # This marks the start of a new block
            net_defs.append({})
            module_defs.append({})
            net_defs[-1]['type']=line[1:-1].rstrip()
            module_defs[-1]['type'] = line[1:-1].rstrip()

            
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0 # in cfg files batch_norm key not present in 1x1 conv layer(before yolo) to add it
                net_defs[-1]['batch_normalize']=0
                c+=1
                
            elif module_defs[-1]['type'] == 'shortcut':
                s+=1
            elif module_defs[-1]['type'] == 'yolo':
                y+=1
            elif module_defs[-1]['type'] == 'route':
                r+=1
            elif module_defs[-1]['type'] == 'upsample':
                u+=1
            else:
                o+=1

    
        
        
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()
            net_defs[-1][key.rstrip()]=value.strip()

    net=darkNet_module_defs.pop(0) # removing 'net' ie hyper parameter dictionary Note: it is not done with net_defs. it will be done later
                                   # during creating modules

    
    print('parsing compleate. layer counts are..')
    print('conv={}, shortcut={}, yolo={},route={},upsample={},others={}'.format(c,s,y,r,u,o))
    
    return (net_defs,darkNet_module_defs,YOLONet_module_defs)


def parse_data_config(path):
    """
        Parses the data configuration file
        
        Args
            path: data configuration file path
        
        Returns
            dictionary containing training options
            
    """
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


if __name__=='__main__':
    path='./config/yolov3-kitti.cfg'
    #parse_model_config(path)
    n,d,y=parse_model_config(path)
    print('yolo=',len(y))
    print('Darknet=',len(d))
    for i in range(len(n)):
        print(n[i])
   
