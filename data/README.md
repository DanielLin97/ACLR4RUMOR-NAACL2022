# Data
The datasets organized and constructed in this work for both cross-domain and cross-lingual experimental settings.  
Twitter: the well-resourced dataset TWITTER for open domains in English  
Weibo: the target low-resource dataset Weibo-COVID19 for COVID-19 domain in Chinese

# Precomputed representations
Running ACLR requires computed XLM-R representations and graph construction in this folder. Warning: these files are quite large. You have two options to generate these:
1. (recommended) download them from:
    https://www.dropbox.com/sh/raz6unw2lswcy54/AADNsc-ifBoAfN1wwyVvgch-a?dl=0
2. You can use the script getWeibograph.py in the folder Process to precompute XLM-R representations for all posts and graph construction for each conversation thread.
