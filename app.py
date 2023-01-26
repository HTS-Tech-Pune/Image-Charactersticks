from flask import Flask, request, jsonify
import base64 
import main
import numpy as np
import cv2
app = Flask(__name__)

'''
Expected Input
{
    img_type: 'EXTENSION'             #the type of image before converting to base64(file extension)
    img: 'BASE64_ENCODED_IMAGE'
    name: 'NAME_OF_THE_PERSON'
    
}

'''

@app.route('/post_json', methods=['POST'])
def process_json():
    json = request.json
    decoded_img = base64.b64decode(json['img'])
    img_type=json['img_type']
    name=json['name']
    # if not os.path.exists(os.getcwd()+"\\dataset\\train\\"+name):
    #     os.mkdir(os.getcwd()+r"\\dataset\\train\\"+name)
        
    # f = open(os.getcwd()+r"\\dataset\train"+"\\"+name+'\\'+name+"."+img_type, "wb")
    # print(type(decoded_img))
    # print(decoded_img)
    
    # f.write(decoded_img)
    # f.close()
    # image = cv2.imread(os.getcwd()+r"\\dataset\train"+"\\"+name+'\\'+name+"."+img_type)
    img = np.frombuffer(decoded_img, np.uint8)
    image = cv2.imdecode(img, cv2.IMREAD_COLOR)
    print(image)
    print(type(image))
    print(image.shape)
    
    result=main.isOkay(image, 50, 50,contrast_thresh= 0.5)
    '''
    result is a dictionary consisting of all the parameres checked and their respective statuses along with a proper reason statement.
    
    result = {'status': bool, 'reasons': suitable message,s None if status is True, parameters_checked = dictionary of all parameters and their respective statuses}
    '''
    # if not result['status']:
    #     os.remove(os.getcwd()+r"\\dataset\train"+"\\"+name+'\\'+name+"."+img_type)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    return jsonify({"result":result})

@app.route('/', methods=['GET'])
def hello_world():
    return "Welcome to the Image verification API"
    
if __name__ == "__main__":
    app.run()