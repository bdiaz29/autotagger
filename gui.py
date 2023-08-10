import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
from glob import glob
tagger=None
include_characters=False
import cv2
import os
import numpy as np
import copy
from tqdm import tqdm
import sqlite3
tags=None
tag2index={}

def tags_no_characters():
    conn = sqlite3.connect('tags.db')
    command = f"SELECT REPLACE(name, '_', ' ') FROM tags WHERE category='0' ORDER BY order_id ASC"
    cur = conn.cursor()
    cur.execute(command)
    rows = cur.fetchall()
    tagout = np.array([row[0] for row in rows])
    return tagout


def tags_with_characters():
    conn = sqlite3.connect('tags.db')
    command = f"SELECT REPLACE(name, '_', ' ') FROM tags WHERE category='0' OR category='4' ORDER BY order_id ASC"
    cur = conn.cursor()
    cur.execute(command)
    rows = cur.fetchall()
    tagout = np.array([row[0] for row in rows])
    return tagout

class ProgressBarHandler:
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def update_progress(self, progress):
        self.progress_bar['value'] = progress
        self.progress_bar.update()

def load_tf_model():
    global tagger
    file_path = filedialog.askdirectory()
    if file_path=='':
        return
    from tfmodel import TFTagger
    try:
        tagger = TFTagger(file_path)
    except Exception as ex:
        error_popup(ex)

def load_trt_model():
    global tagger
    filetypes = [("TRT or Engine files", "*.trt;*.engine")]
    file_path = filedialog.askopenfilename(filetypes=filetypes)
    if file_path=='':
        return

    try:
        from trtmodel import TrtTagger
        tagger = TrtTagger(file_path)
    except Exception as ex:
        error_popup(ex)

def process_probabilities(probabilities, files, threshold, filter,append_to_front):
    global include_characters, tags
    for i, file in enumerate(files):

        probs = probabilities[i]
        spl = file.split('.')
        txt_file = '.'.join(spl[0:-1]) + '.txt'
        file_name = os.path.basename(file)
        if include_characters:
            probs = probs[4:9083] * filter
        else:
            probs = probs[4:6951] * filter
        passedprobs = probs >= threshold
        passedtags = tags[passedprobs]
        passedconfidences = probs[passedprobs]
        sorted_confidecnes = np.argsort(-1 * passedconfidences)
        passedtags = passedtags[sorted_confidecnes]
        appender=copy.copy(append_to_front)
        appender.extend(passedtags)
        outtext = ', '.join(appender)
        wrt = open(txt_file, 'w')
        wrt.write(outtext)
        wrt.close()

def select_directory():
    global progress_handler,tagger,chk_var
    global tags

    if tagger is None:
        error_popup("please load model")
        return

    folder_path = filedialog.askdirectory()
    if folder_path=='':
        return

    include_characters = chk_var.get()
    exclude_tags =entry_tags.get("0.0", "end").replace('\n','').replace(' ','').split(',')
    append_to_front =entry_tags2.get("0.0", "end").replace('\n','').split(',')
    if len(append_to_front)==1:
        if append_to_front[0]=='':
            append_to_front=[]

    if not include_characters:
        tags = tags_no_characters()
    else:
        tags = tags_with_characters()
    L=(len(tags)-1)
    filter = np.ones(len(tags), dtype=np.float32)
    for tag in exclude_tags:
        if tag in tag2index:
            index = tag2index[tag]
            if index<L:
                filter[index] = 0

    try:
        threshold = float(entry_threshold.get())
    except Exception as ex:
        error_popup(ex)
        return

    
    filelist=[]
    filelist.extend(glob(folder_path+'/*.jpg'))
    filelist.extend(glob(folder_path + '/*.png'))
    filelist.extend(glob(folder_path + '/*.jpeg'))
    filelist.extend(glob(folder_path + '/*.webp'))
    total=len(filelist)
    div=100/total



    accumulated_files = []
    img_list = []
    batch_size = tagger.batch_size
    pbar=tqdm(total=len(filelist))
    for i,file in enumerate(filelist):
        pbar.update(1)
        num = int(div * i)
        progress_handler.update_progress(num)
        try:
            img = cv2.imread(file)
        except:
            img = None
        if not img is None:
            img_list += [img]
            accumulated_files += [file]
        if len(accumulated_files) >= batch_size:
            output_probabilities = tagger(img_list)
            process_probabilities(output_probabilities, accumulated_files, threshold, filter, append_to_front)
            accumulated_files = []
            img_list = []

    if len(accumulated_files) > 0:
        Length = len(img_list)
        diff = batch_size - Length
        blank = np.zeros((tagger.height, tagger.width, 3), dtype=np.uint8)
        img_list.extend([blank] * diff)
        output_probabilities = tagger(img_list)[0:Length]
        process_probabilities(output_probabilities, accumulated_files, threshold, filter, append_to_front)
        accumulated_files = []
        img_list = []
        
    progress_handler.update_progress(0)



def error_popup(message):
    messagebox.showerror("Error", message)

ftags=tags_with_characters()
for i, tag in enumerate(ftags):
    tag2index[tag] = i


root = tk.Tk()

btn_load_tf = tk.Button(root, text="Load Tensorflow Model", command=load_tf_model)
btn_load_tf.grid(row=0, column=0, ipadx=5, ipady=5, padx=5, pady=5)

btn_load_trt = tk.Button(root, text="Load Tensor RT Model", command=load_trt_model)
btn_load_trt.grid(row=1, column=0, ipadx=5, ipady=5, padx=5, pady=5)

btn_caption_dir = tk.Button(root, text="Caption Directory", command=select_directory)
btn_caption_dir.grid(row=2, column=0, ipadx=5, ipady=5, padx=5, pady=5)

chk_var = tk.BooleanVar()
chk_include_tags = tk.Checkbutton(root, text="Include character tags",variable=chk_var)
chk_include_tags.grid(row=3, column=0)

lbl_tags = tk.Label(root, text="Tags to exclude (comma separated)")
lbl_tags.grid(row=4, column=0,ipadx=5, ipady=5, padx=5, pady=5)
entry_tags = tk.Text(root, height=3, width=50)
entry_tags.grid(row=4, column=1,ipadx=5, ipady=5, padx=5, pady=5)

lbl_tags2 = tk.Label(root, text="Tags to append (comma separated)")
lbl_tags2.grid(row=5, column=0,ipadx=5, ipady=5, padx=5, pady=5)
entry_tags2 = tk.Text(root, height=3, width=50)
entry_tags2.grid(row=5, column=1,ipadx=5, ipady=5, padx=5, pady=5)

lbl_threshold = tk.Label(root, text="threshold")
lbl_threshold.grid(row=6, column=0,ipadx=5, ipady=5, padx=5, pady=5)
entry_threshold = tk.Entry(root, width=10)
entry_threshold.grid(row=7, column=0,ipadx=5, ipady=5, padx=5, pady=5)
entry_threshold.insert(0, '0.25')

progress_bar = ttk.Progressbar(root, length=650, mode='determinate')
progress_bar.grid(row=8, column=0, columnspan=2)
progress_handler = ProgressBarHandler(progress_bar)

root.geometry("800x600")
root.title("Auto Tagger")
root.mainloop()
