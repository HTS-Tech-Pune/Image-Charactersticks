import cv2
from deepface import DeepFace
from head_detection.head_pose_estimation import head_pose
from is_low_contrast import is_enough_contrast
import mtcnn
import os
from fer import FER


def isOneFace(faces):
    return len(faces)==1

def isSufficient(faces):
    bounding_box=faces[0]['box']
    x, y, w, h=bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
    print("\n",w,h)
    if (w,h)<(100,100):
        return False
    else :
        return True

def isIncomplete(frame,face):
    scale=1
    bounding_box=face['box']
    x, y, w, h=bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
    x1,y1,x2,y2=x,y,x+w,y+h
    p1,p2=(x1-int(w/10),y1-int(h/10)),(x2+int(w/10),y2+int(h/10))
    bounding_box=face['box']
    x, y, w, h=bounding_box[0],bounding_box[1],bounding_box[2],bounding_box[3]
    x1,y1,x2,y2=x,y,x+w,y+h
    p1,p2=(x1-int(w/10),y1-int(h/10)),(x2+int(w/10),y2+int(h/10))
    if p1[0]<0:
        p1=(0,p1[1])
        cv2.rectangle(frame,p1,p2,(0,0,255),2)
        cv2.putText(frame,"Incomplete Face",p2,cv2.FONT_HERSHEY_PLAIN,scale,(0,0,255),2)
        return False
    if p1[1]<0:
        p1=(p1[0],0)
        cv2.rectangle(frame,p1,p2,(0,0,255),2)
        cv2.putText(frame,"Incomplete Face",p2,cv2.FONT_HERSHEY_PLAIN,scale,(0,0,255),2)
        return False
    if p2[0]>frame.shape[1]:
        p2=(frame.shape[1],p2[1])
        cv2.rectangle(frame,p1,p2,(0,0,255),2)
        cv2.putText(frame,"Incomplete Face",p1,cv2.FONT_HERSHEY_PLAIN,scale,(0,0,255),2)
        return False
    if p2[1]>frame.shape[0]:
        p2=(p2[0],frame.shape[0])
        cv2.rectangle(frame,p1,p2,(0,0,255),2)
        cv2.putText(frame,"Incomplete Face",p1,cv2.FONT_HERSHEY_PLAIN,scale,(0,0,255),2)
        return False
    
    cv2.rectangle(frame,p1,p2,(0,155,255),2)
    return True

def isHeadPerfect(frame,thresh1,thresh2):
    ang1,ang2=head_pose(frame)
    return not(ang1>=thresh1 or ang2>=thresh2)

def get_emotions(image):
    if isinstance(image, str):
        image = cv2.imread(image)

    emo_detector = FER(mtcnn=True)
    
    captured_emotions = emo_detector.detect_emotions(image)
    dominant_emotion, emotion_score = emo_detector.top_emotion(image)
    return captured_emotions, dominant_emotion, emotion_score

def isNeutral(image, other_emotions_threshhold =0.75,dominant_emotion_threshhold = 0.5):
    captured_emotions, dominant_emotion, dominant_emotion_score = get_emotions(image)
    if len(captured_emotions):
        if dominant_emotion == 'neutral' or dominant_emotion == 'happy'  and dominant_emotion_score>dominant_emotion_threshhold :
            return True, captured_emotions

        if dominant_emotion == 'angry':
            dominant_emotion_score=captured_emotions[0]['emotions']['neutral']+captured_emotions[0]['emotions']['angry']
            if dominant_emotion_score>dominant_emotion_threshhold and dominant_emotion_score/2 > captured_emotions[0]['emotions']['angry']:
                    return True, captured_emotions
                
        if dominant_emotion == 'sad':
            dominant_emotion_score=captured_emotions[0]['emotions']['neutral']+captured_emotions[0]['emotions']['sad']
            if dominant_emotion_score>dominant_emotion_threshhold and dominant_emotion_score/2 > captured_emotions[0]['emotions']['sad']:
                    return True, captured_emotions
                
        else:
            if captured_emotions[0]['emotions']['neutral'] +captured_emotions[0]['emotions']['angry'] + captured_emotions[0]['emotions']['happy'] + captured_emotions[0]['emotions']['happy'] >=0.8:
                return True, captured_emotions
            
            if dominant_emotion_score > other_emotions_threshhold:
                return False, captured_emotions
           
    return False, [{'box': [132, 89, 153, 209], 'emotions': {'angry': 0.0, 'disgust': 0.0, 'fear': 0.0, 'happy': 0.0, 'sad': 0.0, 'surprise': 0.0, 'neutral': 0.0}}]

def isOkay(frame,thresh1,thresh2, contrast_thresh=0.5):
    
    if isinstance(frame, str):
        frame = cv2.imread(frame)
    emotions=[]
    result={'status':False,'reasons':'','parameters_checked':''}
    parameters_checked=[]
    messages = {'is_one_face': 'Image does not have a single face',
                'is_sufficient':'The image size is not sufficient for analysis. Try giving higher quality image.',
                    'is_incomplete': 'The image is incomplete or has partial face.',
                    'is_head_perfect': 'Head alignment is not proper.',
                    'is_enough_contrast': 'The image has low Contrast', 
                    'is_neutral': 'Please put a neutral expression on face.'}
    
    is_enough_contrast1 = is_enough_contrast(frame,contrast_thresh)
    parameters_checked.append('is_enough_contrast')
    if not is_enough_contrast1:
        return {'status': False,'reasons':messages['is_enough_contrast'],'parameters_checked':parameters_checked}
    
    detector = mtcnn.MTCNN()
    faces = detector.detect_faces(frame)
    
    is_one_face = isOneFace(faces)
    parameters_checked.append('is_one_face')
    if not is_one_face:
        return {'status': False,'reasons':messages['is_one_face'],'parameters_checked':parameters_checked}
    
    is_sufficient=isSufficient(faces)
    parameters_checked.append('is_sufficient')
    if not is_sufficient:
        return {'status': False,'reasons':messages['is_sufficient'],'parameters_checked':parameters_checked}
    
    is_incomplete = isIncomplete(frame,faces[0])
    parameters_checked.append('is_incomplete')
    if not is_incomplete:
        return {'status': False,'reasons':messages['is_incomplete'],'parameters_checked':parameters_checked}
    
    try:
        is_head_perfect = isHeadPerfect(frame,thresh1,thresh2)
    except:
        is_head_perfect = False
    parameters_checked.append('is_head_perfect')
    if not is_head_perfect:
        return {'status': False,'reasons':messages['is_head_perfect'],'parameters_checked':parameters_checked}
    
    is_neutral,emotions = isNeutral(frame)
    parameters_checked.append('is_neutral')
    if not is_neutral:
        return {'status': False,'reasons':messages['is_neutral'],'parameters_checked':parameters_checked}
    
    return {'status': True,'reasons':'','parameters_checked':parameters_checked}
    
    
    # if is_enough_contrast1:
    #     detector = mtcnn.MTCNN()
    #     faces = detector.detect_faces(frame)
    #     is_one_face = isOneFace(faces)
    #     if(is_one_face==False):
    #         is_incomplete=False
    #         is_head_perfect=False
    #         is_neutral=False
    #     else:
    #         is_incomplete = isIncomplete(frame,faces[0])
    #         try:
    #             is_head_perfect = isHeadPerfect(frame,thresh1,thresh2)
    #         except:
    #             is_head_perfect = False
    #         is_neutral,emotions = isNeutral(frame)
        
    #     parameters_checked = {'is_one_face': is_one_face,
    #                             'is_incomplete': is_incomplete,
    #                             'is_head_perfect': is_head_perfect,
    #                             'is_enough_contrast':is_enough_contrast1,
    #                             'is_neutral':is_neutral }
        
    #     if False in parameters_checked.values():
    #         status = False
    #         reasons  = [k for k, v in parameters_checked.items() if v == False]

    #     else:
    #         status = True
    #         reasons = []
    
    #     result = {'status': status,
    #                 'reasons': [messages[i] for i in reasons if reasons],
    #                 'parameters_checked': parameters_checked}
        
    # else:   
    #     parameters_checked = {'is_enough_contrast':is_enough_contrast1}
    #     result = {'status': False,
    #                 'reasons': messages['is_enough_contrast'],
    #                 'parameters_checked': parameters_checked}
    # return result


def run_onPC():
    camera = cv2.VideoCapture(0)
    count=500
    name=input("Enter your name: ")
    while count!=0:
        ret,frame=camera.read()
        frame1=frame.copy()
        thresh1,thresh2=50,50
        # print(isOkay(frame,thresh1,thresh2))
        if (isOkay(frame,thresh1,thresh2)['status']):
            if not os.path.exists("dataset/train/"+name):
                os.mkdir("dataset/train/"+name)
            cv2.imwrite(r"dataset\train"+"\\"+name+'\\'+name+str(count)+".jpg",frame1)
            count-=1
        cv2.imshow('Frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    camera.release()
    cv2.destroyAllWindows()


            
# run_onPC()