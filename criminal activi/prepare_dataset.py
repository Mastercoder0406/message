#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import glob
import pandas as pd
import numpy as np
from moviepy import editor
from tqdm.notebook import tqdm_notebook


# In[2]:


def get_time(timestamp, e=0, max_t=None):
    if e>0 and max_t is None:
        print("Please provide max_t")
        return
    if ":" in timestamp:
        minute = int(timestamp.split(":")[0])*60
        second = float(timestamp.split(":")[1])
        t = minute + second
        return max(t+e, 0) if e<0 else min(t+e, max_t)
    else:
        try:
            return max(float(timestamp)+e, 0) if e<0 else min(float(timestamp)+e, max_t)
        except Exception as e:
            print(e)


# In[3]:


def get_filename(file, n):
    ext = "."+file.split('.')[-1]
    new_name = file[:-len(ext)] + "_" + str(n) + ext
    return new_name


# In[4]:


def extract_clips(data_dir, root="Data", output_root="Processed-Videos", gap=2, skip_classes=[]):
    base_path = os.path.join(root, data_dir)
    if not os.path.exists(base_path):
        print(f"{base_path} does not exists.")
        return
    subdirs = []
    for d in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, d)):
            subdirs.append(d)
    for d in subdirs:
        if d in skip_classes:
            continue
        os.mkdir(os.path.join(root, output_root, d))
        file = open(os.path.join(base_path, f"{d}_Labels.txt"), mode='r', encoding='utf-8')
        lines = file.readlines()
        print(f"Processing {d} category videos:")
        for line in tqdm_notebook(lines):
            l = line.strip().split()
            vid_name = l[0]
            vid_path = os.path.join(base_path, d, vid_name)
            if os.path.exists(vid_path):
                video = editor.VideoFileClip(vid_path)
                if len(l[1:])%2 != 0:
                    print(f"Timestamps are not in complete pairs. Ignoring last timestamp for video {os.path.join(base_path, d, vid_name)}")
                anomaly_timestamps = []
                # Anamoly part extraction
                n_pairs = len(l[1:])//2
                for n in range(n_pairs):
                    new_vid = f"{os.path.join(root, output_root, d, get_filename(vid_name, n))}"
                    t1, t2 = get_time(l[n*2+1], e=-0.7), get_time(l[n*2+2], e=0.7, max_t=video.duration)
                    if t1 is not None and t2 is not None:
                        clip = video.subclip(t1, t2)
                        clip.write_videofile(new_vid, logger=None)
                        anomaly_timestamps.append((t1, t2))
                    else:
                        print(f"Skipping timestamps pairs {l[n*2+1]} and {l[n*2+2]} for {vid_path}")
                # Normal part extraction
                normal_output_dir = os.path.join(root, output_root, "Normal")
                if not os.path.exists(normal_output_dir):
                    os.mkdir(normal_output_dir)
                normal_timestamps = []
                if len(anomaly_timestamps) > 0:
                    if anomaly_timestamps[0][0]-gap > 0:   # gap b/w normal & anomaly
                        normal_timestamps.append((0, anomaly_timestamps[0][0]-gap))
                    for i in range(len(anomaly_timestamps)-1):
                        start = anomaly_timestamps[i][1]+gap
                        end = anomaly_timestamps[i+1][0]-gap
                        if end-start > 0:
                            normal_timestamps.append((start, end))
                    if anomaly_timestamps[-1][1]+gap < video.duration:
                        normal_timestamps.append((anomaly_timestamps[-1][1]+gap, video.duration))
                for i, t in enumerate(normal_timestamps):
                    if t[1]-t[0]>=1.3:
                        clip = video.subclip(t[0], t[1])
                        new_vid = f"{os.path.join(normal_output_dir, get_filename('Normal_'+vid_name, i))}"
                        clip.write_videofile(new_vid, logger=None)
            else:
                print(f"{vid_path} doesnot exists. Skipping...")


# In[9]:


extract_clips("Anomaly-Videos-Part-3")


# In[10]:


extract_clips("Anomaly-Videos-Part-2")


# In[12]:


extract_clips("Anomaly-Videos-Part-2", skip_classes=["Burglary", "Explosion"])


# In[13]:


extract_clips("Anomaly-Videos-Part-1")


# In[ ]:





# ## Splitting Videos into 64 frames clips

# In[19]:


def trim_clips(MAX_SEQ_LENGTH=40, output_dir="Trimmed-Videos", input_dir="Processed-Videos", root="Data"):
    if not os.path.exists(os.path.join(root, output_dir)):
        os.mkdir(os.path.join(root, output_dir))
    for cat in os.listdir(f"{root}/{input_dir}/"):
        os.mkdir(os.path.join(root, output_dir, cat))
        print(f"Processing {cat} category videos:")
        for file in tqdm_notebook(glob.glob(f"{root}/{input_dir}/{cat}/*")):
            video = editor.VideoFileClip(file)
            vid_name = file.split("\\")[-1].strip()
            fps = video.fps
            interval = (1/fps)*MAX_SEQ_LENGTH
            t = 0
            counter = 0
            while (video.duration-t)>= interval:
                clip = video.subclip(t, t+interval)
                clip.write_videofile(f"{root}/{output_dir}/{cat}/{get_filename(vid_name, counter)}", logger=None)
                counter += 1
                t += interval
            if (video.duration-t) > 1.4:
                clip = video.subclip(video.duration-interval, video.duration)
                clip.write_videofile(f"{root}/{output_dir}/{cat}/{get_filename(vid_name, counter)}", logger=None)
            video.close()


# In[20]:


trim_clips(MAX_SEQ_LENGTH=64)


# In[ ]:





# ## Creating dataframe of filenames and their labels

# In[2]:


rooms = []
for item in os.listdir('Data/Trimmed-Videos'):
     files = os.listdir(f'Data/Trimmed-Videos/{item}')

     # Add them to the list
     for file in files:
            rooms.append((item, f'Data/Trimmed-Videos/{item}/{file}'))
    
# Build a dataframe        
df = pd.DataFrame(data=rooms, columns=['label', 'filepath'])
df


# In[3]:


df_normal = df[df['label']=='Normal']
df_normal


# In[4]:


df_anomaly = df[df['label']!='Normal']
df_anomaly


# In[6]:


# Downsampling to balance dataset
drop_indices = np.random.choice(df_normal.index, 5000, replace=False)
df_normal_subset = df_normal.drop(drop_indices)


# In[7]:


df_normal_subset


# In[12]:


df_final = pd.concat([df_normal_subset, df_anomaly])
df_final


# In[13]:


df_final = df_final.reset_index(drop=True)
df_final


# In[15]:


# shuffle the DataFrame rows
df_final = df_final.sample(frac = 1)
df_final


# In[16]:


df_final = df_final.reset_index(drop=True)
df_final


# In[17]:


test_size = 5000


# In[32]:


test_df = df_final.iloc[:5000].reset_index(drop=True)
test_df


# In[33]:


train_df = df_final.iloc[5000:].reset_index(drop=True)
train_df


# In[34]:


train_df.to_csv("Data/train_df.csv", index=False)


# In[35]:


test_df.to_csv("Data/test_df.csv", index=False)


# In[ ]:




