# face-recognizer
## How to run?
### Step 1
In order to run the program, you'll need a few modules. To get them imported, execute:
```
pip install numpy
pip install opencv-python
```
### Step 2
Once you have them, you need a dataset. In order to create a dataset, you'll need some pictures.
Inside the main directory, create another directory called **faces**.    
### Step 3
When you have done that, inside the new directory, create new directories for each person you wanna recognize,
then just copy the pictures of them and put them in their directories.
### Step 4
Once you have your **dataset** ready, you need to train the recognizer, do that by executing:
```
python3 face_recognizer.py
```
### Step 5
Now you're good to go! Just launch the main script by executing 
```
python3 main.py
```
 
### quirks:
_If you want to take a picture of yourself, and place it in your dataset, just press space. The script will then prompt you for your name, and save the picture to your directory._
### future ideas:
_I want to make the model able to tell if the person in front of the camera is wearing a facemask or not_



