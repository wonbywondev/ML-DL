
# 기본 설정
# 1. 경로 설정
import json
import os
import torchvision
import matplotlib.pyplot as plt


ROOT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "train_images")
ANNOT_DIR = os.path.join(DATA_DIR, "train_annotations")

full_dict_path = os.path.join(DATA_DIR, "FULL_DICT.json")
err_txt_path = os.path.join(DATA_DIR, "err_image_paths.txt")
fixed_dict_path = os.path.join(DATA_DIR, "FIXED_DICT.json")
partial_dict_path = os.path.join(DATA_DIR, "PARTIAL_DICT.json")
no_dict_path = os.path.join(DATA_DIR, "NO_DICT.json")



# # 기본 설정
# # 2. 구글 드라이브에서 FIXED_DICT, err_image_paths.txt 최신화

# import gdown
# from datetime import datetime

# err_txt_file_id = "11ZSWLtsCERqCnn3LW8hLkNoabzXMTuF9"
# FIXED_DICT_file_id = "12lXHbkQNowGtk5AxtJZ3638qVLhSwxHh"

# gdown.download(id=err_txt_file_id, output=err_txt_path, quiet=False)
# gdown.download(id=FIXED_DICT_file_id, output=fixed_dict_path, quiet=False)

# err_modification_time = os.path.getmtime(err_txt_path)
# err_modification_time = datetime.fromtimestamp(err_modification_time)

# fixed_modification_time = os.path.getmtime(fixed_dict_path)
# fixed_modification_time = datetime.fromtimestamp(fixed_modification_time)

# print("\n[업데이트 완료]")
# print(f"· err_image_paths 업데이트 날짜: {err_modification_time}")
# print(f"· FIXED_DICT 업데이트 날짜: {fixed_modification_time}")



# 함수 구현

def print_image(image_name):
    image_path = os.path.join(IMAGE_DIR, image_name)
    a = torchvision.io.read_image(image_path).permute(1,2,0)
    plt.imshow(a)
    plt.title(image_name)
    plt.show()


# def get_modi_time(path):
#     modi_time = os.path.getmtime(path)
#     modi_time = datetime.fromtimestamp(modi_time)
#     return modi_time


# 지니 계수(불평등도) 계산
def calculate_gini(dictionary: dict) -> float:
    import numpy as np

    pill_num_count_dict = dict()

    for annot_list in dictionary.values():
        for dic in annot_list:
            pill_num = int(dic["label"])
            try:
                pill_num_count_dict[int(pill_num)] += 1
            except:
                pill_num_count_dict[int(pill_num)] = 1

    counts = list(pill_num_count_dict.values())

    counts = np.array(counts)
    if len(counts) == 0 or np.sum(counts) == 0:
        return 0.0


    sorted_counts = np.sort(counts)
    n = len(counts)
    cum_counts = np.cumsum(sorted_counts)
    gini = (n + 1) / n - 2 * np.sum(cum_counts) / (cum_counts[-1] * n)


    return round(gini, 3)


# FULL_DICT 불러오기
def load_full_dict() -> dict:
    with open(full_dict_path, "r", encoding="utf-8") as f:
        FULL_DICT = json.load(f)
    
    return FULL_DICT



# PARTIAL_DICT 불러오기
def load_partial_dict() -> dict:
    with open(partial_dict_path, "r", encoding="utf-8") as f:
        PARTIAL_DICT = json.load(f)

    return PARTIAL_DICT


# NO_DICT 불러오기
def load_no_dict() -> dict:
    with open(no_dict_path, "r", encoding="utf-8") as f:
        NO_DICT = json.load(f)

    return NO_DICT


# err_image_paths 불러오기
def load_err_image_paths() -> list:
    try:
        with open(err_txt_path, "r", encoding="utf-8") as f:
            err_image_paths = f.read().split()

        return err_image_paths
    
    except:
        pass


# FIXED_DICT 불러오기
def load_fixed_dict() -> dict:
    try:
        with open(fixed_dict_path, "r", encoding="utf-8") as f:
            FIXED_DICT = json.load(f)

        return FIXED_DICT
    except:
        pass


# FINAL_DICT 불러오기
def load_final_dict() -> dict:

    FINAL_DICT = {}

    try:
        FIXED_DICT = load_fixed_dict()

        for image_name, annot_list in FIXED_DICT.items():
            for annot in annot_list:
                del annot["drug_N"]

            FINAL_DICT[os.path.join(IMAGE_DIR, image_name)] = annot_list
        
        
        unique_labels = set()
        for annots in FINAL_DICT.values():
            for annot in annots:
                unique_labels.add(annot["label"])

    except:
        pass

    print(f"<<FINAL_DICT 불러오기 완료>>")
    print(f"· 이미지: {len(FINAL_DICT)}개")
    print(f"· 어노테이션: {sum([len(a) for a in FINAL_DICT.values()])}개")
    print(f"· 클래스: {len(unique_labels)}개")
    print(f"· 지니 계수(불균형도): {calculate_gini(FINAL_DICT)} (낮을수록 균형)")

    return FINAL_DICT


def make_data_yaml():
    import shutil
    from sklearn.model_selection import train_test_split
    from PIL import Image

    FINAL_DICT = load_final_dict()

    # YOLO 데이터셋 폴더 생성
    YOLO_BASE_PATH = "./data/yolo_dataset"
    os.makedirs(f"{YOLO_BASE_PATH}/images/train", exist_ok=True)
    os.makedirs(f"{YOLO_BASE_PATH}/images/val", exist_ok=True)
    os.makedirs(f"{YOLO_BASE_PATH}/labels/train", exist_ok=True)
    os.makedirs(f"{YOLO_BASE_PATH}/labels/val", exist_ok=True)

    # 클래스 ID 매핑 생성 (category id -> 0부터 시작하는 인덱스)
    unique_labels = set()
    for annots in FINAL_DICT.values():
        for annot in annots:
            unique_labels.add(annot["label"])

    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    print(f"총 클래스 수: {len(label_to_idx)}")
    print(f"클래스 매핑: {label_to_idx}")

    # Train/Val 분할 (80:20)
    image_paths = list(FINAL_DICT.keys())
    train_images, val_images = train_test_split(image_paths, test_size=0.2, random_state=42)

    print(f"\nTrain 이미지: {len(train_images)}개")
    print(f"Val 이미지: {len(val_images)}개")

    def convert_to_yolo_format(bbox, img_width, img_height):
        """
        XYXY bbox를 YOLO format (normalized XYWH)으로 변환
        Args:
            bbox: [x1, y1, x2, y2]
            img_width, img_height: 이미지 크기
        Returns:
            [x_center, y_center, width, height] (normalized)
        """
        x, y, w, h = bbox
        
        # 중심점과 너비/높이 계산
        x_center = x + w/2
        y_center = y + h/2
        width = w
        height = h
        
        # 정규화 (0~1 범위)
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]

    def create_yolo_labels(image_paths, split='train'):
        """YOLO 라벨 파일 생성 및 이미지 복사"""
        for img_path in image_paths:
            # 이미지 크기 읽기
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # 이미지 파일명
            img_filename = os.path.basename(img_path)
            img_name = os.path.splitext(img_filename)[0]
            
            # 이미지 복사
            dst_img_path = f"{YOLO_BASE_PATH}/images/{split}/{img_filename}"
            shutil.copy(img_path, dst_img_path)
            
            # 라벨 파일 생성
            label_path = f"{YOLO_BASE_PATH}/labels/{split}/{img_name}.txt"
            
            with open(label_path, 'w') as f:
                annots = FINAL_DICT[img_path]
                for annot in annots:
                    # 클래스 ID 변환
                    class_id = label_to_idx[annot["label"]]
                    
                    # bbox를 YOLO 형식으로 변환
                    yolo_bbox = convert_to_yolo_format(
                        annot["bbox"], 
                        img_width, 
                        img_height
                    )
                    
                    # YOLO 형식으로 작성: <class> <x_center> <y_center> <width> <height>
                    f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

    # Train/Val 라벨 생성
    print("\n라벨 파일 생성 중...")
    create_yolo_labels(train_images, 'train')
    create_yolo_labels(val_images, 'val')
    print("완료!")

    # YOLO data.yaml 파일 생성
    data_yaml = {
        'path': os.path.abspath(YOLO_BASE_PATH),  # 데이터셋 루트 경로
        'train': 'images/train',  # train 이미지 경로
        'val': 'images/val',      # val 이미지 경로
        'nc': len(label_to_idx),  # 클래스 개수
        'names': idx_to_label     # 클래스 이름 (idx: label)
    }

    import yaml
    with open(f"{YOLO_BASE_PATH}/data.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\nYOLO 데이터셋 준비 완료!")
    print(f"경로: {YOLO_BASE_PATH}")
    print(f"data.yaml 생성 완료")
    


def yolo_to_csv():
    import os
    import glob
    import yaml
    import pandas as pd
    from ultralytics import YOLO

    # 1. 경로 설정 (프로젝트 구조에 맞게 조정)
    ROOT_DIR = os.path.dirname(os.getcwd())          # notebooks 기준 한 단계 위
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    TEST_DIR = os.path.join(DATA_DIR, "test_images") # 테스트 이미지 폴더 이름 확인!
    MODEL_PATH = os.path.join(os.getcwd(), "runs", "detect", "pill_y8s_512_aug1_light_with_sampler_no_aug", "weights", "best.pt")
    DATA_YAML_PATH = os.path.join(os.getcwd(), "data", "yolo_dataset", "data.yaml")

    # 2. data.yaml에서 idx -> category_id
    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg["names"]   # {0: 1899, 1: 2482, ...}[file:24]

    # 3. 테스트 이미지 목록 (파일 이름 기준)
    test_image_paths = glob.glob(os.path.join(TEST_DIR, "*.png"))

    # 파일명에서 숫자만 뽑아서 정렬 기준으로 사용
    test_image_paths = sorted(
        test_image_paths,
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))

    # 4. 모델 로드 & 예측
    model = YOLO(MODEL_PATH)
    results = model(test_image_paths, conf=0.25, iou=0.5, verbose=False)

    # 5. 결과 → submission rows
    rows = []
    ann_id = 1

    for img_path, res in zip(test_image_paths, results):
        fname = os.path.basename(img_path)         # "3.png", "10.png" 그대로
        image_id = int(os.path.splitext(fname)[0]) # "3" -> 3, "10" -> 10
        ...
        # 1, 3, 4, ...

        if res.boxes is None or len(res.boxes) == 0:
            # 박스 없는 이미지는 규칙에 따라 처리 (보통 아무 행도 안 넣어도 됨)
            continue

        for box in res.boxes:
            cls_idx = int(box.cls.item())                   # YOLO class index
            category_id = int(names[cls_idx])               # 실제 category_id 숫자[file:24]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            score = float(box.conf.item())

            rows.append({
                "annotation_id": ann_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": x1,
                "bbox_y": y1,
                "bbox_w": w,
                "bbox_h": h,
                "score": score,
            })
            ann_id += 1

    # 6. CSV로 저장
    sub = pd.DataFrame(rows, columns=[
        "annotation_id", "image_id", "category_id",
        "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"
    ])
    print(sub.head())
    sub.to_csv(os.path.join(ROOT_DIR, "submission_test_fixed_with_sampler_no_aug.csv"), index=False)


# def common_to_personal(data):
#     if type(data) == dict:
#         new_dict = {}

#         for key, value in data.items():
#             new_key = os.path.join(IMAGE_DIR, key)
#             new_value_list = []
            
#             for item in value:
#                 new_item = os.path.join(ANNOT_DIR, item)
#                 new_value_list.append(new_item)

#             new_dict[new_key] = new_value_list

#         return new_dict

#     else:
#         for value in data:
#             pass

# def personal_to_common(data):
#     pass
