#! /bin/bash
#pip install -U torch==1.4.0 numpy==1.18.1
#pip install -r requirements.txt
#Generate graph data and store in /data/Weibograph
python ./Process/getWeibograph.py Weibo
#Reproduce the experimental results.
CUDA_VISIBLE_DEVICES=0 python ./model/Weibo/ACLR_Weibo.py Weibo 1
#end
