import os
import sys
import click
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from PIL import Image, ImageOps  # Install pillow instead of PIL
from pathlib import Path
import numpy as np
import subprocess
import shutil
import csv


@click.command()
@click.argument('dir')
def train_model(dir):
    base_model = keras.applications.MobileNet(weights='imagenet', include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # we add dense layers so that the model can learn more complex functions and classify for better results.
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3

    dimensions = len(next(os.walk(dir))[1])  # equals number of directories

    preds = Dense(dimensions, activation='softmax')(x)  # final layer with softmax activation

    model = Model(inputs=base_model.input, outputs=preds)
    for layer in model.layers[:-5]:
        layer.trainable = False
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=keras.applications.mobilenet.preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0, 0.2)
    )

    train_generator = train_datagen.flow_from_directory(
        directory= dir,
        target_size=(224, 224),
        color_mode='rgb',
        batch_size= 1,
        class_mode='categorical',
        shuffle=True
    )

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    step_size_train = train_generator.n // train_generator.batch_size
    model.fit(train_generator, steps_per_epoch=step_size_train, epochs=20)

    model.save('saved_model.keras')

    labels = [f.name for f in os.scandir(dir) if f.is_dir()]

    with open('saved_model/labels.txt', 'w') as f:
        for label in labels:
            f.write(f"{label}\n")



@click.command()
@click.argument('dir')
def split_into_days(dir):
    os.chdir(dir)
    Path('days/').mkdir(parents=True, exist_ok=True)
    week_image_paths = Path('./weeks/').glob('*.jpg')
    
    for image_path in week_image_paths:
        subprocess.run('magick convert ./weeks/' + str(image_path.name) +' -set filename:fn "%t" -crop 5x6@ ./days/%[filename:fn]-%02d.jpg', shell=True)
   

@click.command()
@click.argument('dir')
def predict_days(dir):
    # Load the labels
    with open('saved_model/labels.txt', 'r') as file:
        class_names = file.read().splitlines()
    
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = keras.models.load_model('saved_model.keras', compile=False)

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    os.chdir(dir)
    day_image_paths = Path('./days/').glob('*.jpg')
    for image_path in day_image_paths:
        image = Image.open(image_path).convert("RGB")
        
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        if confidence_score < 0.7:
            class_name = '_unclear'


        # Print prediction and confidence score
        print(image_path)
        print("Class:", class_name)
        print("Confidence Score:", confidence_score)
        
        Path('./classified/' + class_name).mkdir(parents=True, exist_ok=True)
        shutil.move(image_path, './classified/' + class_name + '/' + image_path.name)
        


def create_table(directory):
    table = [["Week", "Weekday", "Person", "Category"]]
    dirs = [x for x in Path(directory + '/classified/').iterdir() if x.is_dir()]
    for dir in dirs:
        for image in Path(dir).iterdir():
            # image.stem is of the format \d\d-\d\d where the first number is the week and the second number is a running number going from left to right, row for row, like reading a book 
            [week, nr] = image.stem.split('-')
            week = int(week)
            nr = int(nr)
            day_of_week = nr % 5
            person = nr // 5
            table.append([week, day_of_week, person, image.parent.name])
    
    with open(directory + '/table.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(table)
    



# if __name__ == "__main__":
#     directory = sys.argv[1]
#     split_into_days(directory)
#     predict_days(directory)
#     input('Check the results for mistakes and move incorrect images into the correct folders. Press enter')
#     create_table(directory)

