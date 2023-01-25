import os
import main
import cv2
import pandas as pd
result=""

test_folder=r"C:\\Users\\nimis_r\\OneDrive\\Desktop\\INTERNSHIPS\\HTS_Internship\\Face_Attendence_System\\Client_Dataset\\train\\"
imgs=os.listdir(test_folder)
with open ("result_train.txt","a") as f:
    for i in imgs[0:10]:
        frame=cv2.imread(test_folder+i)
        res1=i+" : "+str(main.isOkay(frame, 50, 50,contrast_thresh= 0.5))+"\n"
        result+=res1
        print(res1)
        # f.write(res1)
# print(main.isOkay(cv2.imread(test_folder+"zpmHF9aL9qgpWzx59q6PzC0mSYK2.jpg"), 50, 50,contrast_thresh= 0.5))
# df = pd.DataFrame.from_dict(emotions)
# df['image']=imgs