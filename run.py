import cv2
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
features  = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young']


cap=cv2.VideoCapture(0)
while True:
    if cv2.waitKey(2500) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
    
    ret,frame=cap.read()
    model=tf.keras.models.load_model(r'mymodelcelebface.h5')# load model
    if ret==True:
        img = cv2.resize(frame,(170,170)) 
        x = image.img_to_array(img)
        x = np.expand_dims(img, axis=0)

        images = np.vstack([x])
        classes = model.predict(x, batch_size=1)
        classes=classes[0]
       
        top=np.argsort(classes)[-11:][::-1]#print top 10 most confident predication irrespective of gender
        for i in range(1,11):
            perc=str(round((classes[top[i]]*100),2))
            print(str(features[top[i]])+"="+perc+"%")
        
        
    cv2.imshow("output",frame)
    
      
# Break the loop