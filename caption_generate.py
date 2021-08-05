from pickle import load
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
import load_data as ld
import generate_model as gen
#for UI design
from tkinter import *
from tkinter.filedialog import *
from PIL import Image, ImageTk




root = Tk()
root.title("Image Caption Generator") #Title bar
root.configure(background='#81ecec')
w, h = root.winfo_screenwidth(), root.winfo_screenheight() #to maximize the window
root.geometry("%dx%d+0+0" % (w, h))
toolbar=Frame(root, bg='red')
toolbar.pack(fill=X)
Mainframe=Frame(root,bg='#81ecec')
Mainframe.pack()
Topframe=Frame(root,bg='#81ecec')
Topframe.pack()
mytitle=Label(toolbar, text="Image Caption Generator",bg='red',font=("Javanese Text", 18), pady=20) #label for my title
mytitle.pack()
Label1=Label(Mainframe, text ="Select an Image for generating the caption from the imgs folder", bg='#81ecec' , pady=20 , font=("Javanese Text", 14))
Label1.pack()
def selectimage(): #function after clicking select image button
  print ("Rocken")
  for widget in Topframe.winfo_children():
    widget.destroy()

  path=askopenfilename(title="Select an Image",filetypes= (("JEPG files", ".jpg"),("PNG files",".png")))
  print(path) 
  im = Image.open(path)
  re_image=im.resize((550,400))
  tkimage = ImageTk.PhotoImage(re_image)
  myvar=Label(Topframe,image = tkimage)
  myvar.image =tkimage
  myvar.pack()
  G = Button(Topframe,text ="Generate Caption")
  G.config(command =lambda: generate_caption(G,path))
  G.pack()

def generate_caption(button,x):

  button.pack_forget()
  meropath(x)
  

B = Button(Mainframe,text ="Select an Image", command = selectimage)
B.pack()
# t = Button(Mainframe,text ="Sample Train")
# t.pack(side=RIGHT)
#upto here


# extract features from each photo in the directory
def extract_features(filename):
  # load the model
  model = VGG16()
  # re-structure the model
  model.layers.pop() #removing the final softmax layer
  model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
  # load the photo
  image = load_img(filename, target_size=(224, 224))
  # convert the image pixels to a numpy array
  image = img_to_array(image)
  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) #training data,height,width
  # prepare the image for the VGG model
  image = preprocess_input(image)
  # get features
  feature = model.predict(image, verbose=0)
  return feature


# generate a description for an image
def generate_desc(model, tokenizer, photo, index_word, max_length, beam_size=3):

  captions = [['startseq', 0.0]]  #agadi ko start marker and 0.0 is the probability value
  # seed the generation process
  in_text = 'startseq'         
  # iterate over the whole length of the sequence
  for i in range(max_length):     #0 to 34
    all_caps = []				#an array is initialized
    # expand each current candidate
    for cap in captions:      
      sentence, score = cap #sentece word ani score value
      # if final word is 'end' token, just add the current caption
      if sentence.split()[-1] == 'endseq': 
        all_caps.append(cap)
        continue
      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([sentence])[0] #keras function to convert text into sequences, i.e encoding txt2int
      # pad input
      sequence = pad_sequences([sequence], maxlen=max_length) #pad makes the vector size of all elements same and maxlen=34
      # predict next words
      y_pred = model.predict([photo,sequence], verbose=0)[0]  #predict is keras function and for prediction photo feature and text sequences are passed, verbose is used to show the training progress, verbose=0 means showing nothing
      # convert probability to integer
      yhats = np.argsort(y_pred)[-beam_size:]	

      for j in yhats:
        # map integer to word
        word = index_word.get(j)
        # stop if we cannot map the word
        if word is None:
          continue
        # Add word to caption, and generate log prob
        caption = [sentence + ' ' + word, score + np.log(y_pred[j])]
        all_caps.append(caption)

    # order all candidates by score
    ordered = sorted(all_caps, key=lambda tup:tup[1], reverse=True)
    captions = ordered[:beam_size]

  return captions



def meropath(path):
  myfilepath=path
  print ("Function called")

  # load the tokenizer
  tokenizer = load(open('models/tokenizer.pkl', 'rb'))
  index_word = load(open('models/index_word.pkl', 'rb'))
  # pre-define the max sequence length (from training)
  max_length = 34

  #if we use sample weight then uncomment this weightfile
  # filename = 'Forsample/GeneratedData/wholeModel.h5'

  
  #Actual model weight
  filename = 'models/model_weight.h5'
  model = load_model(filename)

  if myfilepath:
    # load and prepare the photograph
    photo = extract_features(myfilepath) #mathi function call vayesi esma image ko feature return hunx
    # generate description
    captions = generate_desc(model, tokenizer, photo, index_word, max_length) #training file, caption tokenized file, photo feature extracted, index to word, maximumlenght=34 and receive the caption
    for cap in captions: 
      # remove start and end tokens
      seq = cap[0].split()[1:-1] #breaking down the caption again into arrays and remove the first word start and last word end
      desc = ' '.join(seq)

      #my_producedcaption='{} '.format(desc,cap[1])
      #print (cap[1])

      my_producedcaption='{} [log probability: {:1.2f}]'.format(desc,cap[1])
      # rock.printcaptions(my_producedcaption)
      #printing caption on GUI
      label2=Label(Topframe, text =my_producedcaption,bg='#81ecec',font=("Javanese Text", 16))
      label2.pack()

      print('{} [log prob: {:1.2f}]'.format(desc,cap[1]))
 



root.mainloop()