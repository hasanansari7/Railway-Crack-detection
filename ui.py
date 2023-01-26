import tkinter as tk     ##tkinter is package is a gui 
from tkinter.filedialog import askopenfilename  ##
import shutil
import os
import sys
from PIL import Image, ImageTk
import cv2



window = tk.Tk()

window.title("Dr. Detection")

window.geometry("800x910")
window.configure(background ="yellow")

title = tk.Label(text="Click below to choose picture for testing track....", background = "lightgreen", fg="Brown", font=("", 15))
title.grid()
IMG_SIZE = 50
def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    print("path " + verify_dir)
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'Railwaytrackcrack-{}-{}.model'.format(LR, '2conv-basic')
##    MODEL_NAME='keras_model.h5'
    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (int(IMG_SIZE), int(IMG_SIZE)))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data
##    def Send():
##        data.write(str.encode('C'))
##        print('Sent the Character')

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
    tf.compat.v1.reset_default_graph()
    #tf.reset_default_graph()

    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 128, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 32, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = conv_2d(convnet, 64, 3, activation='relu')
    convnet = max_pool_2d(convnet, 3)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))


        if np.argmax(model_out) == 0:
            str_label = 'Broken'
            print('Broken')
        elif np.argmax(model_out) == 1:
            str_label = 'Non defective'
            print('Non defective')

        if str_label == 'Broken':
            status= '   Found Crack on Track   '
            labelb = tk.Label(text='STATUS : ' + status, background="darkcyan",
                               fg="Red", font=("", 15))
            labelb.grid(column=0, row=4, padx=20, pady=20)
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)

            
        elif str_label == 'Non defective':
            status = "No Defects Found on track"
##            stages()
            labelb = tk.Label(text='STATUS: ' + status, background="darkcyan",
                               fg="Black", font=("", 15))
            labelb.grid(column=0, row=4, padx=20, pady=20)
            button = tk.Button(text="Exit", command=exit)
            button.grid(column=0, row=9, padx=20, pady=20)

        button1 = tk.Button(text="open new photo", activebackground="red", command = openphoto)
        button1.grid(column=0, row=2, padx=10, pady = 10)
        

def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    # C:/Users/sagpa/Downloads/images is the location of the image which you want to test..... you can change it according to the image location you have  
    fileName = askopenfilename(initialdir='C:\\Users\\hasan\\OneDrive\\Desktop\\Railway crack track detection\\updated_proj\\test', title='Select image for analysis ',
                           filetypes=[('image files', '.jpg')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    if os.path.split(fileName)[-1].split('.') == 'h (1)':
        print('dfdffffffffffffff')



        
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
##    image = cv2.imread('b (9).jpg')
##    render = cv2.resize(render, (IMG_SIZE, IMG_SIZE) ) 
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="500", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="  Analyse Image  ", activebackground="red", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", activebackground="red",command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)



window.mainloop()



