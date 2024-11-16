from metro.tools.demo_bstro import main
import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
import joblib
import pyvista as pv
import trimesh
import cv2
from PIL import Image
from glob import glob

def parse_args():
    parser = ArgumentParser()
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,128', type=str, 
                        help="The Image Feature Dimension.")   
    parser.add_argument("--legacy_setting", default=True, action='store_true',)
    
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--config_name", default="", type=str, 
                        help="Pretrained config name or path if not the same as model_name.")
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument("--output_attentions", default=False, action='store_true',) 

    parser.add_argument("--dataset", default="EMDB2")


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    DATA_ROOT = f"/bigdata/zhizheng/data/josh_demo" # f"/data1_zhizheng/zhizheng/{args.dataset}"
    OUTPUT_ROOT = args.dataset
    for si, scene in enumerate(sorted(os.listdir(DATA_ROOT))):
        scene_path = os.path.join(DATA_ROOT, scene)
        scene_files = os.listdir(os.path.join(scene_path))

        if args.dataset == "EMDB2":
            for file in scene_files:
                if file[-3:] == "pkl" and scene in file:
                    scene_data_path = file
        elif args.dataset == "SLOPER4D":
            scene_data_path = glob(os.path.join(scene_path, "*_labels.pkl"))[0]
        else:
            # for in-the-wild videos
            scene_data_path = os.path.join(scene_path, "tram", "hps")
        
        if not scene_data_path:
            print(f"Can't find scene data file for {scene}")
            continue

        all_bboxes = {}
        all_frames = {}
        if args.dataset == "EMDB2":
            scene_data = joblib.load(os.path.join(scene_path, scene_data_path))
            # bboxes = scene_data['bboxes']['bboxes']
            all_bboxes[0] = scene_data['bboxes']['bboxes']
        elif args.dataset == "SLOPER4D":
            from scripts.sloper4d_dataset import SLOPER4D_Dataset
            scene_data = SLOPER4D_Dataset(os.path.join(scene_path, scene_data_path))
            # bboxes = scene_data.bbox
            all_bboxes[0] = scene_data.bbox
        else:
            for i, track in enumerate(os.listdir(scene_data_path)):
                scene_data = np.load(os.path.join(scene_data_path, track), allow_pickle=True)
                scene_data = scene_data.item()
                all_bboxes[i] = scene_data["bbox"][:, :4].int().numpy()
                all_frames[i] = scene_data["frame"]

        os.makedirs(os.path.join(OUTPUT_ROOT, scene), exist_ok=True)

        image_files = []
        if args.dataset == "EMDB2":
            image_files = sorted(os.listdir(os.path.join(scene_path, "images")))
        elif args.dataset == "SLOPER4D":
            image_files = scene_data.file_basename
        else:
            all_images = sorted(os.listdir(os.path.join(scene_path, "images")))

        for track_id in all_bboxes:
            bboxes = all_bboxes[track_id]
            if args.dataset == "wild":
                image_files = [file for i, file in enumerate(all_images) if i in all_frames[track_id]]

            track_id = str(track_id)
            os.makedirs(os.path.join(OUTPUT_ROOT, scene, track_id), exist_ok=True)

            for i, image in tqdm(enumerate(image_files)):
                if os.path.exists(os.path.join(OUTPUT_ROOT, scene, track_id, image.replace("jpg", "npy"))):
                    continue
                # (x_min, y_min, x_max, y_max)
                curr_bbox = bboxes[i]
                rgb = Image.open(os.path.join(scene_path, "images", image))
                if len(curr_bbox) > 0:
                    rgb = rgb.crop(tuple(list(curr_bbox))) # rgb.crop(tuple(curr_bbox.tolist()))
                rgb.save(f"demo/temp_{image}")
                args.input_img = os.path.join(scene_path, "images", image)
                args.resume_checkpoint = "models/bstro/hsi_hrnet_3dpw_b32_checkpoint_15.bin"
                pred_contact, pred_contact_meshes = main(args)
                pred_contact = pred_contact[0].cpu().numpy()
                pred_contact_meshes = np.array(pred_contact_meshes[0].vertices)
                pred = np.hstack([pred_contact_meshes, pred_contact])

                pv.start_xvfb()
                mesh = pv.read("demo/contact_vis.ply")
                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(mesh)

                center = mesh.center
                plotter.camera_position = [
                    (center[0], center[1] - 2, center[2] + 5),
                    center,
                    (0, 1, 0)
                ]
                plotter.screenshot(os.path.join(OUTPUT_ROOT, scene, track_id, f"{image[:-4]}_front.png"))
                plotter.close()

                plotter = pv.Plotter(off_screen=True)
                plotter.add_mesh(mesh)
                plotter.camera_position = [
                    (center[0], center[1] - 2, center[2] - 5),
                    center,
                    (0, 1, 0)
                ]
                plotter.screenshot(os.path.join(OUTPUT_ROOT, scene, track_id, f"{image[:-4]}_back.png"))
                plotter.close()

                rgb = cv2.imread(f"demo/temp_{image}")
                cv2.imwrite(os.path.join(OUTPUT_ROOT, scene, track_id, image), rgb)
                # import ipdb; ipdb.set_trace()

                np.save(os.path.join(OUTPUT_ROOT, scene, track_id, image.replace("jpg", "npy")), pred)

                os.remove(f"demo/temp_{image}")