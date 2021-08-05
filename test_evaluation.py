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



# generate a description for an image
def generate_desc(model, tokenizer, photo, index_word, max_length, beam_size=5):

  captions = [['startseq', 0.0]]
  # seed the generation process
  in_text = 'startseq'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    all_caps = []
    # expand each current candidate
    for cap in captions:
      sentence, score = cap
      # if final word is 'end' token, just add the current caption
      if sentence.split()[-1] == 'endseq':
        all_caps.append(cap)
        continue
      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([sentence])[0]
      # pad input
      sequence = pad_sequences([sequence], maxlen=max_length)
      # predict next words
      y_pred = model.predict([photo,sequence], verbose=0)[0]
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

# evaluate the skill of the model
def evaluate_model(model, descriptions, photos, tokenizer, index_word, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in descriptions.items():
    # generate description
    yhat = generate_desc(model, tokenizer, photos[key], index_word, max_length)[0]
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    # Use best caption
    predicted.append(yhat[0].split())
  # calculate BLEU score
  print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
  print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
  print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
  print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

if __name__ == '__main__':
	# load the tokenizer
  tokenizer = load(open('models/tokenizer.pkl', 'rb'))
  index_word = load(open('models/index_word.pkl', 'rb'))
  # pre-define the max sequence length (from training)
  max_length = 34

  # load the model
  filename = 'models/model_weight.h5'
  model = load_model(filename)
  test_features, test_descriptions = ld.prepare_dataset('test')[1]
  # evaluate model
  evaluate_model(model, test_descriptions, test_features, tokenizer, index_word, max_length)