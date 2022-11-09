from imp import init_builtin
from symbol import factor
import cv2
import numpy as np
import os
import warnings
from datetime import datetime as dt

from ying import Ying_2017_CAIP
from LIME import LIME as lime
# from simplyLIME import simplyLIME as lime
from dhe import dhe

warnings.filterwarnings("ignore")

def adjust_gamma_table(gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return table

def brighten(root_path="/media/mldadmin/home/s122mdg36_01/caipei/assignment1/dataset", init_dirname='train', is_gamma=True, method='Ying_CAIP',edge_method='canny', gamma=1.0, print_timespan=False):

    method_path = os.path.join(root_path, '%s_%s_%s' % (init_dirname, method,edge_method))

    if not os.path.exists(method_path):
        os.mkdir(method_path)

    if is_gamma:
        transform = adjust_gamma_table(gamma)
    else:
        gamma = 1.0

    with open(os.path.join(root_path, "%s.txt" % init_dirname)) as f:
        lines = f.readlines()
    videos = [line.strip().split('\t') for line in lines]

    i=0
    for name in videos:
        if len(name[-1].split('/')) == 2:
            os.makedirs(os.path.join(method_path, name[-1].split('/')[0]), exist_ok=True)            
        print("Start time: {}".format(dt.now()))

        # capture the video frame by frame
        cap = cv2.VideoCapture(os.path.join(root_path, "%s_%s" % (init_dirname,method), name[-1]))
        length_pre = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap_writer = cv2.VideoWriter(os.path.join(method_path, name[-1]), cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
        print("save_path: %s" % os.path.join(method_path, name[-1]))
        begin = dt.now()
        j=0
        while(True):
            ret, frame = cap.read()
            
            if ret == True:
                # if is_gamma or method.upper() == 'GAMMA':
                #     new_frame = cv2.LUT(frame, LU_table)
                # else:
                #     if method.upper() == 'YING_CAIP':
                #         new_frame = Ying_2017_CAIP(frame)
                #     elif method.upper() == 'DHE':
                #         new_frame = dhe(frame)
                #     elif method.upper() == 'LIME':
                #         new_frame = lime(frame, print_process=False)
                #         new_frame = new_frame.enhance()
                #     elif method.upper() == 'CANNY':
                #         new_frame = cv2.Canny(frame,200,300)
                        
                if edge_method.upper() == 'CANNY_50_150':
                    new_frame = cv2.Canny(frame,100,200)
                    new_frame = np.expand_dims(new_frame, axis=2)
                    new_frame = np.concatenate((new_frame, new_frame, new_frame), axis=-1)
      
                j+=1          
                cap_writer.write(new_frame)
            
            else:
                #print("Completed the processing of %s" %(name[-1]))
                end = dt.now()
                if print_timespan:
                    span = (end - begin).total_seconds()
                    #print("Video {} takes {} seconds to convert".format(name[-1].split('/')[-1], span))
                break

        cv2.destroyAllWindows()     # close all the widows opened inside the program
        cap.release        			# release the video read/write handler
        cap_writer.release
        i+=1
        # if i>10:
        #     break

    print("End time: {}".format(dt.now()))

if __name__ == '__main__':
    # brighten(root_path='/media/mldadmin/home/s122mdg36_01/caipei/assignment1/dataset', init_dirname='train', is_gamma=False, method='lime', edge_method='addcanny',print_timespan=True)
    brighten(root_path='../', init_dirname='test', is_gamma=False, method='lime',edge_method='canny_50_150', print_timespan=True)
