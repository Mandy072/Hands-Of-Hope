from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from app.verify import authentication
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import cache_control
from .add_sign import *
from .training import *
from .forms import HandSignForm
from .models import HandImage, SentenceToVideo
from django.core.files.base import ContentFile
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import shutil
from django.conf import settings
import pyttsx3
import tempfile
import imageio
import speech_recognition as sr
# from app.process import xray_prediction


def record_and_transcribe():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Please speak...")  # Prompt the user to speak
        audio_data = recognizer.listen(source)  # Record the audio input

    try:
        print("Recognizing...")  # Indicate that recognition is in progress
        # Use Google Speech Recognition to transcribe the audio
        text = recognizer.recognize_google(audio_data)
        print("You said:", text)  # Print the transcribed text
        return text

    except sr.UnknownValueError:
        print("Sorry, could not understand audio.")  # Handle unrecognized speech
        return None

    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

def pronounce(sentence):
    engine = pyttsx3.init()
    engine.say(sentence)
    engine.runAndWait()

# Function to preprocess hand landmarks
def preprocess_hand_landmarks(hand_landmarks):
    landmarks = [point['x'] for point in hand_landmarks['hand_landmarks']]
    return np.array(landmarks)


##############################################################################
#                               Main Section                                 #
##############################################################################


def index(request):
    # return HttpResponse("This is Home page")    
    return render(request, "index.html")

def log_in(request):
    if request.method == "POST":
        # return HttpResponse("This is Home page")  
        username = request.POST['username']
        password = request.POST['password']

        user = authenticate(username = username, password = password)

        if user is not None:
            login(request, user)
            messages.success(request, "Log In Successful...!")
            return redirect("dashboard")
        else:
            messages.error(request, "Invalid User...!")
            return redirect("log_in")
    # return HttpResponse("This is Home page")    
    return render(request, "log_in.html")

def register(request):
    if request.method == "POST":
        fname = request.POST['fname']
        lname = request.POST['lname']
        username = request.POST['username']
        password = request.POST['password']
        password1 = request.POST['password1']
        # print(fname, contact_no, ussername)
        verify = authentication(fname, lname, password, password1)
        if verify == "success":
            user = User.objects.create_user(username, password, password1)          #create_user
            user.first_name = fname
            user.last_name = lname
            user.save()
            messages.success(request, "Your Account has been Created.")
            return redirect("/")
            
        else:
            messages.error(request, verify)
            return redirect("register")
    # return HttpResponse("This is Home page")    
    return render(request, "register.html")


@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def log_out(request):
    logout(request)
    messages.success(request, "Log out Successfuly...!")
    return redirect("/")

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def dashboard(request):
    context = {
        'fname': request.user.first_name, 
        
        }
    return render(request, "dashboard.html",context)

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def new_sign(request):
    sign_data = HandImage.objects.all()
    if request.method == "POST":
        form = HandSignForm(request.POST)
        if form.is_valid():
            new_sign = form.cleaned_data['new_sign']
            new_sign = str(new_sign).lower()
            last_image_path = add_sign(new_sign)

            if last_image_path is not None:
                # Save the image in the database
                hand_image = HandImage(sign_name=new_sign)

                # Read the image file
                with open(last_image_path, "rb") as image_file:
                    # Save the image to the Django model
                    hand_image.image.save(f"{new_sign}.jpg", ContentFile(image_file.read()), save=True)

                train_verify = train_classes()
                if train_verify == "Success":
                    messages.success(request, "Data Added and Trained Succesfully..!!!")
                else:
                    messages.error(request, train_verify)
                return redirect('dashboard') 
    else:
        form = HandSignForm()

    context = {
        'fname': request.user.first_name, 
        'form': form,
        'sign_data' : sign_data
    }

    return render(request, "new_sign.html", context)

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def capture_sign(request):
    sign_data = HandImage.objects.all()
    if request.method == "POST":
         # Load the trained model
        model = tf.keras.models.load_model('Dataset/sign_language_model.h5',compile=False)

        # Initialize Mediapipe Hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands()

        cap = cv2.VideoCapture(0)

        # Set the margin for the bounding box
        margin = 20
        predicted_names = ["0"]
        while True:
            _, frame = cap.read()

            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image with Mediapipe Hands
            results = hands.process(rgb_frame)

            # Check if hands are detected
            if results.multi_hand_landmarks:
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract bounding box coordinates with margin
                    bbox_min = [float('inf'), float('inf')]  # Initialize with large values
                    bbox_max = [float('-inf'), float('-inf')]  # Initialize with small values

                    for landmark in hand_landmarks.landmark:
                        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

                        # Update bounding box coordinates
                        bbox_min[0] = min(bbox_min[0], x)
                        bbox_min[1] = min(bbox_min[1], y)
                        bbox_max[0] = max(bbox_max[0], x)
                        bbox_max[1] = max(bbox_max[1], y)

                    # Add margin to the bounding box
                    bbox_min[0] = max(0, bbox_min[0] - margin)
                    bbox_min[1] = max(0, bbox_min[1] - margin)
                    bbox_max[0] = min(frame.shape[1], bbox_max[0] + margin)
                    bbox_max[1] = min(frame.shape[0], bbox_max[1] + margin)

                    # Draw bounding box
                    cv2.rectangle(frame, (int(bbox_min[0]), int(bbox_min[1])),
                                (int(bbox_max[0]), int(bbox_max[1])), (0, 255, 0), 2)

                    # Capture hand landmarks
                    hand_landmarks_data = {'hand_landmarks': []}
                    for landmark in hand_landmarks.landmark:
                        hand_landmarks_data['hand_landmarks'].append({'x': landmark.x, 'y': landmark.y, 'z': landmark.z})

                    # Preprocess hand landmarks
                    hand_landmarks_processed = preprocess_hand_landmarks(hand_landmarks_data)

                    # Resize the image to match the input size expected by the model
                    hand_landmarks_resized = np.expand_dims(hand_landmarks_processed, axis=0)

                    # Make prediction
                    prediction = model.predict(hand_landmarks_resized)
                    predicted_class = np.argmax(prediction)
                    print(predicted_class)
                    with open('Dataset/label_mapping.json', 'r') as json_file:
                        label_mapping = json.load(json_file)

                    
                    # Map the predicted class to its corresponding label
                    if str(predicted_class) in label_mapping:
                        predicted_label = label_mapping[str(predicted_class)]
                        predicted_class_name = label_mapping.get(str(predicted_class), "Unknown")
                        # Extract only the last part of the path (class name)
                        predicted_class_name = os.path.basename(predicted_label)
                        # Display the predicted sign
                        cv2.putText(frame, f"Predicted Sign: {predicted_class_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        if predicted_class != predicted_names[len(predicted_names)-1]:
                            print(predicted_names[len(predicted_names)-1])
                            print(predicted_class_name)
                            pronounce(predicted_class_name)
                            
                            predicted_names.append(predicted_class_name)
                        
                    else:
                        cv2.putText(frame, "Label not found for the predicted class.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    # Display the result
                    cv2.imshow("Sign Prediction", frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture and close all windows
        cap.release()
        cv2.destroyAllWindows()

    context = {
        'fname': request.user.first_name, 
        'sign_data' : sign_data
    }
    return render(request, "capture_sign.html", context)

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def delete(request, sign_name):
    try:
        # Get the HandImage instance with the given sign_name
        hand_image_instance = HandImage.objects.get(sign_name=sign_name)
        # Get the path to the folder associated with the sign_name
        folder_path = os.path.join("Dataset/Signs", sign_name)
        # Delete the HandImage instance
        hand_image_instance.delete()
        # Remove the folder associated with the sign_name
        if os.path.exists(folder_path):
            # Use shutil.rmtree() to remove the directory and its contents recursively
            shutil.rmtree(folder_path)

        # Update label_mapping.json
        label_mapping_path = 'Dataset/label_mapping.json'
        with open(label_mapping_path, 'r') as label_mapping_file:
            label_mapping = json.load(label_mapping_file)
        # Remove the entry corresponding to the sign_name
        for key, value in label_mapping.items():
            if value == sign_name:
                del label_mapping[key]
                break  # Stop iteration after first match

        with open(label_mapping_path, 'w') as label_mapping_file:
            json.dump(label_mapping, label_mapping_file)

        # Update labels_info.json
        labels_info_path = 'Dataset/labels_info.json'
        with open(labels_info_path, 'r') as labels_info_file:
            labels_info = json.load(labels_info_file)
        # Remove the entry corresponding to the sign_name
        for key, value in labels_info.items():
            if value == sign_name:
                del labels_info[key]
                break  # Stop iteration after first match

        with open(labels_info_path, 'w') as labels_info_file:
            json.dump(labels_info, labels_info_file)
        
        train_classes()

    except HandImage.DoesNotExist:
        pass

    return redirect("new_sign")

@login_required(login_url="log_in")
@cache_control(no_cache = True, must_revalidate = True, no_store = True)
def learn_sign(request):
    sentence_data = SentenceToVideo.objects.all()
    context = {
        'fname': request.user.first_name, 
        'sentence': sentence_data
    }
    
    if request.method == "POST":
        if 'text' in request.POST:
            sentence = request.POST['sentence'].lower()
        else:
            sentence = str(record_and_transcribe()).lower()

        # Tokenize the input sentence into words
        input_words = sentence.split()  
        dataset_words = [str(word.sign_name) for word in HandImage.objects.all()]

        # Match words from dataset with input words
        images_folder = os.path.join(settings.MEDIA_ROOT, 'hand_images')
        matched_images = []
        for word in input_words:
            image_path = os.path.join(images_folder, word + '.jpg')
            if os.path.exists(image_path):
                matched_images.append(image_path)
        
        print("Matched images:", matched_images)

        # Generate video from matched images
        if matched_images:
            frame_width, frame_height, fps = 640, 420, 1
            video_folder = os.path.join(settings.MEDIA_ROOT, 'videos')
            os.makedirs(video_folder, exist_ok=True)

            try:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.flush()

                writer = imageio.get_writer(temp_path, fps=fps, format='FFMPEG')

                for image_path in matched_images:
                    img = cv2.imread(image_path)
                    if img is None:
                        raise Exception(f"Error reading image: {image_path}")

                    img = cv2.resize(img, (frame_width, frame_height))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    writer.append_data(img)

                writer.close()
                print(f"Video saved at: {temp_path}")

                # Move or copy the video to the final location
                video_filename = sentence.replace(" ", "_") + '.mp4'
                final_video_path = os.path.join(video_folder, video_filename)
                shutil.copy(temp_path, final_video_path)
                print(f"Copied video to: {final_video_path}")

                # Store relative path in DB
                relative_video_path = os.path.join('videos', video_filename)
                SentenceToVideo.objects.create(sentence=sentence, video=relative_video_path)

                context['video_url'] = os.path.join(settings.MEDIA_URL, relative_video_path)
                messages.success(request, "Video Generated Successfully!!")
                return redirect("learn_sign")

            except Exception as e:
                messages.error(request, f"Error: {str(e)}")
                return redirect("learn_sign")

    return render(request, "learn_sign.html", context)