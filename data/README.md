# Data
The datasets organized and constructed in this work for both cross-domain and cross-lingual experimental settings.  
Twitter: the well-resourced dataset TWITTER for open domains in English  
Weibo: the target low-resource dataset Weibo-COVID19 for COVID-19 domain in Chinese

Consecutive columns correspond to the following pieces of information:  
1: root-id -- an unique identifier describing the tree/conversation (id of the root);  
2: index-of-parent-post -- an index number of the parent post for the current post;  
3: index-of-the-current-post -- an index number of the current post;  
4: time-delay -- the time slot between the current post and the root;  
5: text -- the textual content of the current post.

# Precomputed representations
Running ACLR requires computed XLM-R representations and graph construction in this folder. Warning: these files are quite large. You have two options to generate these:
1. (recommended) download them from:
    https://drive.google.com/drive/folders/1gvuSeorLAljGZaD7gyWrUA0gyotT_rl6?usp=sharing;
    or https://1drv.ms/f/s!Ar3GISsxjxJPiMlJvXXSLwMD9xrDPA;  
    or https://www.dropbox.com/sh/raz6unw2lswcy54/AADNsc-ifBoAfN1wwyVvgch-a?dl=0
2. You can use the script getWeibograph.py in the folder Process to precompute XLM-R representations for all posts and graph construction for each conversation thread.
