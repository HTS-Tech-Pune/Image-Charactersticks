from flask import Flask, request, jsonify
import base64 
import main
import cv2
import os

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
    if not os.path.exists(os.getcwd()+"\\dataset\\train\\"+name):
        os.mkdir(os.getcwd()+r"\\dataset\\train\\"+name)
        
    f = open(os.getcwd()+r"\\dataset\train"+"\\"+name+'\\'+name+"."+img_type, "wb")
    
    f.write(decoded_img)
    f.close()
    image = cv2.imread(os.getcwd()+r"\\dataset\train"+"\\"+name+'\\'+name+"."+img_type)
    
    result=main.isOkay(image, 50, 50,contrast_thresh= 0.5)
    '''
    result is a dictionary consisting of all the parameres checked and their respective statuses along with a proper reason statement.
    
    result = {'status': bool, 'reasons': suitable message,s None if status is True, parameters_checked = dictionary of all parameters and their respective statuses}
    '''
    if not result['status']:
        os.remove(os.getcwd()+r"\\dataset\train"+"\\"+name+'\\'+name+"."+img_type)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    return jsonify({"result":result})
    
if __name__ == "__main__":
    # res = main.isOkay(r'dataset\train\Neemeesh\Neemeesh.png', 50, 50,contrast_thresh= 0.5)
    # print(res)
    app.run(debug = True)