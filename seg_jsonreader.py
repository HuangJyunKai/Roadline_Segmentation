#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:21:17 2019

@author: huangjunkai
"""

# {sw : 人行道, dr : 可行駛道路區域, ud : 不可行駛道路區域}
# {zc ; 斑馬線, cl : 網黃線, rs : 紅線, ys : 黃線實線, yd : 黃線虛線}
# {ds : 雙黃線實線, dd : 雙黃線虛線, ws : 白線實線, wd : 白線虛線}
# {pm : 機車格, pc : 汽車格, Pb : 公車格, pw : 機車等待區, pl : 機車待轉區, pp : 停車格角點框}

import json
import cv2
import numpy as np
import os
# cv2 (B,G,R)
color_full = {'sw' : ( 0 , 0 , 0), 'dr' : (255,255, 0 ), 'ud' : (0, 255 , 0 ),
              'zc' : ( 0 , 0 , 0), 'cl' : (255, 0, 0), 'rs' : ( 0 , 0 ,255), 'ys' : ( 0 ,255,255), 'yd' : (102,255,255),
              'ds' : ( 0 ,204,255), 'dd' : (102,204,255), 'ws' : ( 0 , 0 , 0 ), 'wd' : (102,153,153),
              'pm' : (255,102, 0 ), 'pc' : ( 51,255,102), 'pb' : (255, 51,153), 'pw' : (153,204,255), 'pl' : (204,255,255), 'pp': (102, 0 ,255)}
line_class = {'zc' : 5, 'cl' : 6 ,'rs' : 7 , 'ys' : 8, 'yd' : 8,
              'ds' : 9, 'dd' : 9 ,'ws' : 10, 'wd' : 11,
              'pm' : 1, 'pc' : 2,'pb' : 3, 'pw' : 4,'pl' : 4, 'pp': 0}
road_class = {'sw' : 1, 'dr' : 2 ,'ud' : 0 }            
line = ['zc','cl','rs','ys','yd','ds','dd','ws','wd','pm','pc','pb','pw','pl']
road = ['sw','dr','ud']
'''
pallete_line = [ [255, 0, 255], [128, 0, 128], [0, 64, 64], [0, 0, 0], [0, 0, 0], \
            [255, 0, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 255, 0], \
            [255, 0, 255], [0,0,0] ]
pallete_road= [ [0,0,0], [255, 255, 0], [0, 255, 0]]

# labels_line={'機車格':0,
#              '汽車格':1,
#              '公車格':2,
#              '機車等待區':3,'機車待轉區':3,
#              '斑馬線':4,
#              '網黃線':5,
#              '紅線':6,
#              '黃線':7,'黃線':7,
#              '雙黃線':8,'雙黃線':8,
#              '白線 實線':9,
#              '白線 虛線':10}
#              'Background':11}

# road segmentation
# pallete = [ [0, 0, 0], [255, 0, 0], [0, 255, 0] ]
# labels_seg={'人行道':1,
#             '可行駛道路區域':2,
#             'Ignore':0}
'''
    
def writepng(file,path,filename):
    with open(file,"r",encoding="utf-8") as f:
        data = json.load(f)
        alpha = 0.05
        beta = 1-alpha
        gamma=0
        filejpg = file.replace('.json','.jpg')
        image = cv2.imread(filejpg)
        overlay = image.copy()
        output = image.copy()
        imgbackground = np.zeros((1080, 1920, 3))
        imgbackground.fill(0)
        imgbackgroundc = imgbackground.copy()
        imgbackground_r = np.zeros((1080, 1920, 3))
        imgbackground_r.fill(0)
        imgbackground_rc = imgbackground.copy()
        linezero = np.zeros((1080, 1920))
        roadzero = np.zeros((1080, 1920))
        checkline = False
        checkroad = False
        for i in range(len(data['shapes'])):
            '''
            print("item:",i)
            print("label:",data['shapes'][i]['label'])
            print("line_color:",data['shapes'][i]['line_color'])
            print("fill_color:",data['shapes'][i]['fill_color'])
            print("points:",data['shapes'][i]['points'])
            print("shape_type:",data['shapes'][i]['shape_type'])
            '''
            '''
            if data['shapes'][i]['shape_type'] == 'rectangle':
                cv2.rectangle(overlay, (int(data['shapes'][i]['points'][0][0]), int(data['shapes'][i]['points'][0][1])), 
                                     (int(data['shapes'][i]['points'][1][0]), int(data['shapes'][i]['points'][1][1])),
                                      (0, 255, 0), -1) 
                cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
            '''
            # 車道線與格線
            if data['shapes'][i]['label'] in line:
                pts = np.array(data['shapes'][i]['points'])
                #cv2.fillPoly(overlay, np.array([pts], dtype=np.int32), color_full[data['shapes'][i]['label']])
                #cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
                #cv2.fillPoly(imgbackgroundc, np.array([pts], dtype=np.int32), color_full[data['shapes'][i]['label']])
                #cv2.addWeighted(imgbackgroundc, alpha, imgbackground, 1 - alpha,0, imgbackground)
                cv2.fillPoly(linezero, np.array([pts], dtype=np.int32), line_class[data['shapes'][i]['label']])
                checkline = True


            # 路跟非路
            '''
            elif data['shapes'][i]['label'] in road: 
                pts = np.array(data['shapes'][i]['points'])
                #cv2.fillPoly(overlay, np.array([pts], dtype=np.int32), color_full[data['shapes'][i]['label']])
                #cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)
                cv2.fillPoly(imgbackground_rc, np.array([pts], dtype=np.int32), color_full[data['shapes'][i]['label']])
                cv2.addWeighted(imgbackground_rc, alpha, imgbackground_r, 1 - alpha,0, imgbackground_r)
                cv2.fillPoly(roadzero, np.array([pts], dtype=np.int32), road_class[data['shapes'][i]['label']])
                checkroad = True
            '''
            
        outfile = filename.replace('.json','')
        print(outfile)
        #cv2.imwrite(path + outfile +'_bgr_background.png', output)
        if checkline :
            #cv2.imwrite(path + outfile +'_bgr_line.png', imgbackground)
            cv2.imwrite(path + outfile +'_lineclass.png', linezero)
        elif checkroad:
            #cv2.imwrite(path + outfile +'_bgr_road.png', imgbackground_r)
            cv2.imwrite(path + outfile +'_roadclass.png', roadzero)

if __name__ == '__main__':
    path = "./OV_001-1-linelabel/"
    if not os.path.isdir(path):
        os.mkdir(path)
    #root = ["./OV_001-Part2_OV_001_0001/"
            #,"./OV_001-Part3_OV_001_0001/","./OV_002-Part2_OV_002_0001/",
            #"./OV_001-Part2_OV_001_0001-Segmentation/",
            #"./OV_001-Part3_OV_001_0001-Segmentation/","./OV_002-Part2_OV_002_0001-Segmentation/"
    #       ]
    root = ["./OV_001-1/"]
    for roots in root:
        print("open :",roots)
        for filename in os.listdir(roots):
            if '.json' in filename:
                print(filename)
                writepng(roots+filename,path,filename)

            

