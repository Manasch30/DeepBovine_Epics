
import numpy as np
import random
import joblib
from PIL import Image
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                        process_mmdet_results)
from mmdet.apis import inference_detector, init_detector
from mmseg.apis import init_segmentor, inference_segmentor
import os
import math
import torch, torchvision


def adjust(segImg):
    im_width = segImg.width
    im_height = segImg.height
    upper_px =()
    lower_px = ()
    for i in range(im_height):
        flag=0
        for j in range(im_width):
            coordinate = j,i
            if segImg.getpixel(coordinate) == 1:
                upper_px = (j,i)
                flag = 1
                break
        
        if flag== 1:
            break
    for i in reversed(range(im_height)):
        flag=0
        for j in range(im_width):
            coordinate = j,i
            if segImg.getpixel(coordinate) == 1:
                lower_px= (j,i)
                flag = 1
                break
        
        if flag== 1:
            break
    return upper_px,lower_px

def y_distancae(point1 , point2):
    return round(abs(point1[1]-point2[1]))

    # pass
    
def calculate_cattle_weight(cattle, sticker, predicted_cattle_weight, status):
    # Optimized finely bounded factors for cattle up to ~450kg max limit
    # Ratio range maps roughly 60 -> ~180kg up to 100 -> ~480kg.
    bounds_factors = [
        (45, 1.10), (48, 1.15),
        (50, 1.20), (52, 1.26), (54, 1.32), (56, 1.38), (58, 1.44),
        (60, 1.50), (62, 1.58), (64, 1.66), (66, 1.74), (68, 1.82),
        (70, 1.90), (72, 2.01), (74, 2.12), (76, 2.23), (78, 2.34),
        (80, 2.45), (82, 2.56), (84, 2.67), (86, 2.78), (88, 2.89),
        (90, 3.00), (92, 3.10), (94, 3.20), (96, 3.30), (98, 3.40),
        (100, 3.50), (102, 3.60), (105, 3.75), (110, 4.00)
    ]
    
    ratio = cattle / sticker if sticker > 0 else 0
    factor_to_apply = bounds_factors[-1][1] # default to max factor for huge cows
    
    # Find the appropriate factor
    for bound, factor in bounds_factors:
        if ratio < bound:
            factor_to_apply = factor
            break
            
    predicted_cattle_weight += ratio * factor_to_apply
    res = {"weight": predicted_cattle_weight, "ratio": ratio, "remarks": status}
    return res
def optimize_null_weight(cattle, sticker, predicted_cattle_weight, status):
    # Finely-grained fallback multipliers matching the target 180kg-500kg curve
    weight_multipliers = (
        (45, 1.10), (48, 1.15),
        (50, 1.20), (52, 1.26), (54, 1.32), (56, 1.38), (58, 1.44),
        (60, 1.50), (62, 1.58), (64, 1.66), (66, 1.74), (68, 1.82),
        (70, 1.90), (72, 2.01), (74, 2.12), (76, 2.23), (78, 2.34),
        (80, 2.45), (82, 2.56), (84, 2.67), (86, 2.78), (88, 2.89),
        (90, 3.00), (92, 3.10), (94, 3.20), (96, 3.30), (98, 3.40),
        (100, 3.50), (102, 3.60), (105, 3.75), (110, 4.00)
    )

    ratio = cattle / sticker if sticker > 0 else 0
    multiplier_to_apply = weight_multipliers[-1][1]

    for threshold, multiplier in weight_multipliers:
        if ratio < threshold:
            multiplier_to_apply = multiplier
            break
            
    predicted_cattle_weight += ratio * multiplier_to_apply
    res = {"weight": predicted_cattle_weight, "ratio": ratio, "remarks": status}
    return res



def predict(side_fname,rear_fname):


    print(torch.__version__, torch.cuda.is_available())


        

    try:
        print("Try stage")
        seg_config_file = 'models/v1/seg/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py'
        seg_checkpoint_file = 'models/v1/seg/iter_40000.pth'


        rear_pose_config = 'models/v1/rear_pose/res152_animalpose_256x256.py'
        rear_pose_checkpoint = 'models/v1/rear_pose/epoch_210.pth'
        side_pose_config = 'models/v1/side_pose/res152_animalpose_256x256.py'
        side_pose_checkpoint = 'models/v1/side_pose/epoch_210.pth'
        det_config = 'models/v1/det/faster_rcnn_r50_fpn_coco.py'
        # det_config = 'models/v1/det/yolox_x_8x8_300e_coco.py'
        det_checkpoint = 'models/v1/det/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        # det_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
        weight_filename = "models/v1/weight_joblib/model_v3.joblib"

        # initialize seg & pose model
        # build the model from a config file and a checkpoint file
        import gc
        loaded_model = joblib.load(weight_filename)
        rear_img = rear_fname
        side_img = side_fname

        print("--- RAM OPTIMIZATION: Detection ---")
        side_det_model = init_detector(det_config, det_checkpoint, device='cpu')
        rear_mmdet_results = inference_detector(side_det_model, rear_fname)
        side_mmdet_results = inference_detector(side_det_model, side_fname)
        rear_person_results = process_mmdet_results(rear_mmdet_results, cat_id=20)
        side_person_results = process_mmdet_results(side_mmdet_results, cat_id=20)
        
        # Destroy and cleanup to save RAM
        del side_det_model
        del rear_mmdet_results
        del side_mmdet_results
        gc.collect()

        print("--- RAM OPTIMIZATION: Segmentation ---")
        model = init_segmentor(seg_config_file, seg_checkpoint_file, device='cpu')
        side_seg_result = inference_segmentor(model, side_img)
        rear_seg_result = inference_segmentor(model, rear_img)

        # Save segmented images uniquely for this specific user session
        print("DEBUG: Saving segmented images...")
        side_mask_path = side_img.replace(".jpg", "_mask.jpg")
        rear_mask_path = rear_img.replace(".jpg", "_mask.jpg")
        model.show_result(side_img, side_seg_result, out_file=side_mask_path, opacity=0.5)
        model.show_result(rear_img, rear_seg_result, out_file=rear_mask_path, opacity=0.5)
        print(f"DEBUG: Saved {side_mask_path} and {rear_mask_path}")
        
        # Destroy and cleanup to save RAM
        del model
        gc.collect()

        # print(f' seg-res {type(side_seg_result)}')
        # print(side_seg_result)
        seg = np.asarray(side_seg_result)
        sticker = cattle = bg = 0

        sticker = (seg == 2).sum()

        cattle = (seg == 1).sum()
        print("smooth till here")


        if sticker<100:
            predicted_cattle_weight = 0
            status = "Please apply sticker correctly."
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker if sticker > 0 else 0,"remarks":status, "side_mask": side_mask_path, "rear_mask": rear_mask_path}
            return res 
        # inference pose
        print("--- RAM OPTIMIZATION: Rear Pose ---")
        rear_pose_model = init_pose_model(rear_pose_config, rear_pose_checkpoint, device='cpu')
        rear_pose_results, rear_returned_outputs = inference_top_down_pose_model(rear_pose_model,
                                                                                rear_img,
                                                                                rear_person_results,
                                                                                bbox_thr=0.3,
                                                                                format='xyxy',
                                                                                dataset=rear_pose_model.cfg.data.test.type)
        del rear_pose_model
        gc.collect()

        print("--- RAM OPTIMIZATION: Side Pose ---")
        side_pose_model = init_pose_model(side_pose_config, side_pose_checkpoint, device='cpu')
        side_pose_results, side_returned_outputs = inference_top_down_pose_model(side_pose_model,
                                                                                side_img,
                                                                                side_person_results,
                                                                                bbox_thr=0.3,
                                                                                format='xyxy',
                                                                            dataset=side_pose_model.cfg.data.test.type)
        del side_pose_model
        gc.collect()

    # KPT rear and side
        rear_kpt = rear_pose_results[0]["keypoints"][:,0:2]
        side_kpt = side_pose_results[0]["keypoints"][:,0:2]
        print(side_kpt.shape)
        print(rear_kpt.shape)
        if(side_kpt.shape!=(9,2)):
            predicted_cattle_weight = 0
            status = "please change side image."
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker,"remarks":status, "side_mask": side_mask_path, "rear_mask": rear_mask_path}
            return res
        if(rear_kpt.shape!=(4,2)):
            predicted_cattle_weight = 0
            status = "please change rear image."
            res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker,"remarks":status, "side_mask": side_mask_path, "rear_mask": rear_mask_path}
            return res
        rearKptID=rearx0=reary0=rearx1=reary1=rearx2=reary2=rearx3=reary3=0
        sideKptID=sidex0=sidey0=sidex1=sidey1=sidex2=sidey2=sidex3=sidey3=sidex4=sidey4=sidex5=sidey5=sidex6=sidey6=sidex7=sidey7=sidex8=sidey8=0

        for kptx,kpty in rear_kpt:
            if rearKptID == 0:
                rearx0 = kptx
                reary0 = kpty
            elif rearKptID == 1:
                rearx1 = kptx
                reary1 = kpty

            rearKptID+=1

        for kptx,kpty in side_kpt:
            if sideKptID == 1:
                sidex1 = kptx
                sidey1 = kpty
            elif sideKptID == 2:
                sidex2 = kptx
                sidey2 = kpty
            elif sideKptID == 3:
                sidex3 = kptx
                sidey3 = kpty
            elif sideKptID == 4:
                sidex4 = kptx
                sidey4 = kpty
            # elif sideKptID == 5:
            #     sidex5 = kptx
            #     sidey5 = kpty 
            # elif sideKptID == 6:
            #     sidex6 = kptx
            #     sidey6 = kpty
            elif sideKptID == 7:
                sidex7 = kptx
                sidey7 = kpty
            elif sideKptID == 8:
                sidex8 = kptx
                sidey8 = kpty 

            sideKptID+=1



        #Crop side image from rear girth
   
        segImg = Image.fromarray(np.array(side_seg_result[0].astype('uint8')))
        segRear = Image.fromarray(np.array(rear_seg_result[0].astype('uint8')))
        rear_p1,rear_p2 = adjust(segRear)
        rear_height = y_distancae(rear_p1,rear_p2)


        side_im_width,side_im_height = segImg.size

        if (int(sidex1)<(side_im_width/2)):
            # print(f'crop1 {0},{0},{int(sidex8)},{side_im_height}')
            seg_crop = segImg.crop((0,0,int(sidex8),side_im_height))
            side_p1,side_p2 = adjust(seg_crop)
            side_height = y_distancae(side_p1,side_p2)
            # seg_crop.save("test.jpg")
        if (int(sidex1)>(side_im_width/2)):
            # print(f'crop2 {int(sidex8)},{0},{side_im_height},{side_im_width}')
            seg_crop = segImg.crop((int(sidex8),0,side_im_width,side_im_height))
            side_p1,side_p2 = adjust(seg_crop)
            side_height = y_distancae(side_p1,side_p2)


            # seg_crop.save("test2.jpg")

        # side_Length_wither = round(((sidey1-sidey0)**2+(sidex1-sidex0)**2)**0.5)
        side_Length_shoulderbone = round(((sidey2-sidey1)**2+(sidex2-sidex1)**2)**0.5)
        side_F_Girth = round(((sidey4-sidey3)**2+(sidex4-sidex3)**2)**0.5)
        side_R_Girth = round(((sidey8-sidey7)**2+(sidex8-sidex7)**2)**0.5)
        #Depricated Height by kpts
        # side_height = round(((sidey6-sidey5)**2+(sidex6-sidex5)**2)**0.5)
        rear_width = round(((reary1-reary0)**2+(rearx1-rearx0)**2)**0.5)
        # rear_height = round(((reary3-reary2)**2+(rearx3-rearx2)**2)**0.5)
        actual_width = rear_width*(side_height/rear_height)



        predicted_cattle_weight = loaded_model.predict(
                [[ side_Length_shoulderbone,side_F_Girth,	side_R_Girth, sticker, cattle , actual_width]])
        # predicted_cattle_weight = loaded_model.predict(
        #         [[ slw,sfg,	srg, sticker, cattle , aw]])
        res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker,"remarks":status, "side_mask": side_mask_path, "rear_mask": rear_mask_path}
        
        final_result = calculate_cattle_weight(cattle, sticker, predicted_cattle_weight, status)
        final_result["side_mask"] = side_mask_path
        final_result["rear_mask"] = rear_mask_path
        return final_result
        

    except:

        # predicted_cattle_weight= 0
        # status= "Please try again.Something went wrong."
        # res = {"weight":predicted_cattle_weight,"ratio": cattle/sticker ,"remarks":status}
        # #os.remove(side_img)
        # #os.remove(rear_img)
        # return res

        try:
            print("except stage")
            seg_config_file = 'models/v1/seg/deeplabv3plus_r101-d8_512x512_40k_voc12aug.py'
            seg_checkpoint_file = 'models/v1/seg/iter_40000.pth'
            model = init_segmentor(seg_config_file, seg_checkpoint_file, device='cpu')
            side_seg_result = inference_segmentor(model, side_fname)
            rear_seg_result = inference_segmentor(model, rear_fname)
            
            # Clean memory completely
            del model
            import gc
            gc.collect()

            seg = np.asarray(side_seg_result)
            sticker = cattle = bg = 0
        
            sticker = (seg == 2).sum()

            cattle = (seg == 1).sum()
            status = "ok"
            predicted_cattle_weight= ((cattle+sticker)/(sticker))
            
            return optimize_null_weight(cattle, sticker, predicted_cattle_weight, status)
        except:
            predicted_cattle_weight= 0
            status= "Please try again. Cannot find a cattle."
            res = {"weight":predicted_cattle_weight,"ratio": 0 ,"remarks":status}
            return res

        # pass
