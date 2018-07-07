# Voice_Style_Transfer

The goal of this project is to extract and transfer meaningful voice features across audio samples without losing the content information of the audio sample.
Attempts to perform voice transfer, inspired by Gatys et al.'s work in image domain. Instead of using a content loss and a style loss, we use perceptual loss from two pretrained networks in speaker recognition and Speech-to-text Wavenet.

The code for the pretrained networks :
Wavenet : https://github.com/buriburisuri/speech-to-text-wavenet <br>
Speaker Recognition : https://github.com/Akella17/Speaker-Recognition

While the perceptual loss extracted from Speech-to-text network preserves content of an audio sample, the loss from Speaker recognition network preserves the voice features. As style transfer performs gradient ascent over the input, we preferred the Speech-to-text network architecture built from dilated convolutions over stacked LSTM to ease training.
<br>
To read more about neural style tranfer and perceptual loss refer these links : <br>
https://arxiv.org/abs/1508.06576 <br>
https://arxiv.org/abs/1603.08155
