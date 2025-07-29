from django.shortcuts import render,HttpResponse


from django.shortcuts import render,redirect
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import login ,logout,authenticate
from django.core.files.storage import default_storage
import tempfile
def home(request):
    return render(request,'Home.html')

def register(request):
    if request.method == 'POST':
        First_Name = request.POST['name']
        Email = request.POST['email']
        username = request.POST['username']
        password = request.POST['password']
        confirmation_password = request.POST['cnfm_password']
        select_user=request.POST['role']
        if select_user=='admin':
            admin=True
        else:
            admin=False
        if password == confirmation_password:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists, please choose a different one.')
                return redirect('register')
            else:
                if User.objects.filter(email=Email).exists():
                    messages.error(request, 'Email already exists, please choose a different one.')
                    return redirect('register')
                else:
                    user = User.objects.create_user(
                        username=username,
                        password=password,
                        email=Email,
                        first_name=First_Name,
                        is_staff=admin
                    )
                    user.save()
                    return redirect('login')
        else:
            messages.error(request, 'Passwords do not match.')
        return render(request, 'register.html')
    return render(request, 'register.html')

def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        if User.objects.filter(username=username).exists():
            user=User.objects.get(username=username)
            if user.check_password(password):
                user = authenticate(username=username,password=password)
                if user is not None:
                    login(request,user)
                    messages.success(request,'login successfull')
                    return redirect('/')
                else:
                   messages.error(request,'please check the Password Properly')
                   return redirect('login')
            else:
                messages.error(request,"please check the Password Properly")  
                return redirect('login') 
        else:
            messages.error(request,"username doesn't exist")
            return redirect('login')
    return render(request,'login.html')
# Load and preprocess the dataset
def logout_view(request):
    logout(request)
    return redirect('login')


# Ignore all warnings
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib
matplotlib.use('Agg')
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from skimage import io, transform
from sklearn import preprocessing
import numpy as np
import joblib
import cv2

path = r"Dataset"
model_folder = "model"
categories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
categories
X_file = os.path.join(model_folder, "X.txt.npy")
Y_file = os.path.join(model_folder, "Y.txt.npy")
if os.path.exists(X_file) and os.path.exists(Y_file):
    X = np.load(X_file)
    Y = np.load(Y_file)
    print("X and Y arrays loaded successfully.")
else:
    X = [] # input array
    Y = [] # output array
    for root, dirs, directory in os.walk(path):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(f'Loading category: {dirs}')
            print(name+" "+root+"/"+directory[j])
            if 'Thumbs.db' not in directory[j]:
                img_array = cv2.imread(root+"/"+directory[j])
                img_resized = resize(img_array, (64, 64, 3))
                # Append the input image array to X
                X.append(img_resized.flatten())
                # Append the index of the category in categories list to Y
                Y.append(categories.index(name))
    X = np.array(X)
    Y = np.array(Y)
    np.save(X_file, X)
    np.save(Y_file, Y)
    
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=77)
labels=categories

precision = []
recall = []
fscore = []
accuracy = []
#function to calculate various metrics such as accuracy, precision etc
def calculateMetrics(plot_path,algorithm, predict, testY):
    testY = testY.astype('int')
    predict = predict.astype('int')
    p = precision_score(testY, predict, average='macro') * 100
    r = recall_score(testY, predict, average='macro') * 100
    f = f1_score(testY, predict, average='macro') * 100
    a = accuracy_score(testY, predict) * 100 
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    
    print(f"{algorithm} Accuracy    : {a}")
    print(f"{algorithm} Precision   : {p}")
    print(f"{algorithm} Recall      : {r}")
    print(f"{algorithm} FSCORE      : {f}")
    
    report = classification_report(predict, testY, target_names=labels)
    print(f"\n{algorithm} classification report\n{report}")
    
    conf_matrix = confusion_matrix(testY, predict)
    plt.figure(figsize=(5, 5))
    
    # Plot the heatmap with reversed axis labels
    ax = sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="Reds", fmt="g")
    
    # Change the labels on the axes to indicate the reverse
    plt.title(f"{algorithm} Confusion matrix")
    plt.xlabel("True class")     # <--- Changed xlabel to "True class"
    plt.ylabel("Predicted class") # <--- Changed ylabel to "Predicted class"
    plt.savefig(plot_path)   
    plt.close() 



from sklearn.tree import DecisionTreeClassifier
import os
import joblib

# Path for the model file
def DTC_existing(request):
    Model_file = os.path.join(model_folder, "DT_Model.pkl")

    # Check if the pkl file exists
    if os.path.exists(Model_file):
        # Load the model from the pkl file
        dt_classifier = joblib.load(Model_file)
        predict = dt_classifier.predict(x_test)
        image='static/images/dtc.png'
        calculateMetrics(image,"DecisionTreeClassifier", predict, y_test)
    else:
        # Create a Decision Tree Classifier
        dt_classifier = DecisionTreeClassifier(max_depth=2,min_samples_split=2)
        dt_classifier.fit(x_train, y_train)
        # Save the model weights to a pkl file
        joblib.dump(dt_classifier, Model_file)  
        predict = dt_classifier.predict(x_test)
        image='static/images/dtc.png'
        print("Decision Tree model trained and model weights saved.")
        calculateMetrics(image,"DecisionTreeClassifier", predict, y_test)
    return render(request,'prediction.html',
                  {'algorithm':'Decision Tree Classifier',
                   'image':image,
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})
from sklearn.ensemble import RandomForestClassifier
def RFC(request):
    # Check if the pkl file exists
    Model_file = os.path.join(model_folder, "RFC_Model.pkl")
    if os.path.exists(Model_file):
        # Load the model from the pkl file
        rf_classifier = joblib.load(Model_file)
        predict = rf_classifier.predict(x_test)
        image='static/images/rfc.png'
        calculateMetrics(image,"RandomForestClassifier", predict, y_test)
    else:
        
        # Create Random Forest Classifier with Decision Tree as base estimator
        rf_classifier = RandomForestClassifier(n_estimators=2)
        rf_classifier.fit(x_train, y_train)
        # Save the model weights to a pkl file
        joblib.dump(rf_classifier, Model_file)  
        predict = rf_classifier.predict(x_test)
        image='static/images/rfc.png'
        print("Random Forest model trained and model weights saved.")
        calculateMetrics(image,"RandomForestClassifier", predict, y_test)
    return render(request,'prediction.html',
                  {'algorithm':'Random Forest Classifier',
                    'image':image,
                   'accuracy':accuracy[-1],
                   'precision':precision[-1],
                   'recall':recall[-1],
                   'fscore':fscore[-1]})


import matplotlib.pyplot as plt
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
import seaborn as sns
import os
import cv2
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

model=model_folder
num_classes=2

def CNN(request):
    import os
    import pickle
    from tensorflow.keras.models import Sequential, model_from_json, load_model
    from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.initializers import glorot_uniform

    # Define model folder and files
    model_folder = 'model'
    Model_file = os.path.join(model_folder, "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel_weights.h5")
    Model_history = os.path.join(model_folder, "history.pckl")

    # Check if model file exists
    if os.path.exists(Model_file):
        # Load model architecture and weights with custom initializer
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json, custom_objects={'GlorotUniform': glorot_uniform})
        model.load_weights(Model_weights)
        print(model.summary())
        
        # Load training history
        with open(Model_history, 'rb') as f:
            history = pickle.load(f)

        # Handle accuracy key dynamically
        acc = history.get('accuracy', history.get('acc', [0]*20))[18] * 100
        
        print("CNN Model Prediction Accuracy = {:.2f}%".format(acc))
    else:
        # Create and train new model
        model = Sequential()
        model.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=num_classes, activation='softmax'))
        
        model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())

        # Ensure X_train and X_test are in the correct shape: (num_samples, 64, 64, 3)
        X_train = x_train.reshape(-1, 64, 64, 3)
        X_test = x_test.reshape(-1, 64, 64, 3)
        
        # Train the model
        hist = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test), shuffle=True, verbose=2)

        # Save model weights and architecture
        model.save_weights(Model_weights)
        model_json = model.to_json()
        with open(Model_file, "w") as json_file:
            json_file.write(model_json)
        
        # Save training history
        with open(Model_history, 'wb') as f:
            pickle.dump(hist.history, f)

        # Handle accuracy key dynamically for training history
        acc = hist.history.get('accuracy', hist.history.get('acc', [0]*20))[18] * 100
        
        print("CNN Model Prediction Accuracy = {:.2f}%".format(acc))

    return render(request, 'prediction.html',
                  {'algorithm': 'CNN Model Classifier',
                   'accuracy': acc,
                   'precision': precision[-1],
                   'recall': recall[-1],
                   'fscore': fscore[-1]})
def prediction_view(request):
    import os
    import cv2
    import numpy as np
    from django.core.files.storage import default_storage
    from tensorflow.keras.models import model_from_json
    from tensorflow.keras.initializers import glorot_uniform
    import matplotlib.pyplot as plt

    model_folder = 'model'
    Model_file = os.path.join(model_folder, "DLmodel.json")
    Model_weights = os.path.join(model_folder, "DLmodel_weights.h5")

    if request.method == 'POST' and request.FILES.get('file'):
        # Load model architecture and weights
        with open(Model_file, "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json, custom_objects={'GlorotUniform': glorot_uniform})
        model.load_weights(Model_weights)

        uploaded_file = request.FILES['file']
        file_path = default_storage.save(uploaded_file.name, uploaded_file)
        
        img = cv2.imread(file_path)
        img_resized = cv2.resize(img, (64, 64))  # Resize image to the target size
        img_array = np.array(img_resized)        # Convert to NumPy array
        img_reshaped = img_array.reshape(1, 64, 64, 3)  # Reshape for model input
        # Normalize the image
        img_normalized = img_reshaped.astype('float32') / 255.0

        # Predict using the model
        pred_probability = model.predict(img_normalized)
        pred_number = np.argmax(pred_probability)  # Get the index of the highest probability

        
        output_name = categories[pred_number]
        default_storage.delete(file_path)

        # Display the image with prediction
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct display in matplotlib
        plt.text(10, 10, f'Predicted Output: {output_name}', color='white', fontsize=12, weight='bold', backgroundcolor='black')
        plt.axis('off')
        image='static/images/output.png'
        plt.savefig(image)
        plt.close()
        return render(request, 'prediction.html', {'predict':image})

    return render(request, 'prediction.html', {'test': True})
