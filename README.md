# ai_sum_between_absolute_1000
 
 This project allow the next:
 Is configurated with two inputs, if you create a csv file 'dataset' with three columns (the two inputs and the label), called by default 'a','b' and 'c', the index will train the AI with an architectura, which you can change if you want.

 Also, you will see how differents graphic plot will appear, showing:
 - The weights
 - The bias
 - The neural net estructure
 - The loss per epoch

 For them you will need to install graphviz

 Finally, when the training is finished, you will see the tipical summary and some more data.
 Your model will be saved on the rute .model/name.h5 on hfd5 format. You can change the name obviously.

 On the file testing.py you can test trained models with new data (for now is reading it from code, will be more clean if imports another csv file).
 And the file compare_models.py can compare diferent models that you have trained. It will show a graph comparation with loss and mae values between models.

 Warning: This is a project made for a newbie on python and AI so you will find a lot of mistakes or ways to improve the code. Feel free to comment. 
