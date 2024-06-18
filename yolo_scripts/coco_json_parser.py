"""
This file contains several useful functions: 
    
    unpack_and_unite_zips - The general function of unpacking and concatenating all images from their zip files + 
                            concatenating all instances_default.json after markup in COCO JSON format

    unpack_zips - Unpacking zip archives: all pictures are copied to the output_folder/images folder, 
                  all coco json files are copied to the annotations folder

    unite_jsons_from_list - Generic handler for concatenating json files in a folder

    save_dataset_for_detector - Generates train/val/test jsons

    get_json_statistics - json file stats

    show_annotations - Saves images with visual markup
"""


import os
import re
import cv2
import json
import shutil
import zipfile
import numpy as np
import xml.etree.ElementTree as ElementTree

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from detectron2.utils.visualizer import GenericMask, Visualizer, ColorMode

JSON_HEADER = {
    "licenses": [
        {
            "name": "",
            "id": 0,
            "url": ""
        }
    ],
    "info": {
        "contributor": "",
        "date_created": "",
        "description": "",
        "url": "",
        "version": "",
        "year": ""
    }
}

coco_categories = ["sliding", "swing", "roll-up"] #TODO IF ERROR PUT IT INTO LINE 60

class CamerasDataParser:
    def __init__(self):
        pass

    @staticmethod
    def unpack_and_unite_zips(path_to_zips_folder: str, path_to_unpack_folder: str):
        """Общая функция распаковки и объединения всех картинок их zip файлов + объединение всеx instances_default.json после разметки в формате COCO JSON
        Пример зип файла после разметки:
            01_batch.zip >>>
                /annotations/
                    instances_default.json
                /images/
                    ...
                    2292_2_2022-07-17.05-54-23.066861.jpg
                    2512_0_2022-07-17.05-55-35.252871.jpg
                    2981_1_2022-07-17.05-38-25.480580.jpg
                    ...

        Args:
            path_to_zips_folder (str): Путь к папке с зипами
            path_to_unpack_folder (str): Путь куда распаковывать

        """
        path_to_annotations, path_to_images = CamerasDataParser.unpack_zips(path_to_zips_folder, path_to_unpack_folder)
        united_json_data = CamerasDataParser.unite_jsons_in_folder(path_to_annotations=path_to_annotations,
                                                                   path_to_images=path_to_images,
                                                                   cats=coco_categories,
                                                                   save_filename='united_annotations.json')

        return united_json_data

    @staticmethod
    def unpack_zips(path_to_zips_folder: str, path_to_unpack_folder: str):
        """Распаковка zip архивов: все картинки копируются в папку output_folder/images, все coco json файлы в папку annotations

        Args:
            path_to_zips_folder (str): Путь к папке с зипами
            path_to_unpack_folder (str): Путь куда распаковывать

        Returns:
            (str, str): Путь к папке с json файлами | Путь к папке с картинками 
        """
        print('...Unzip started...')

        if not os.path.exists(path_to_unpack_folder):
            os.mkdir(path_to_unpack_folder)

        annotations_dir_name = 'annotations'
        images_dir_name = 'images'

        # create separate dirs for annotations (.json) and images (.jpg)
        if not os.path.exists(os.path.join(path_to_unpack_folder, annotations_dir_name)):
            os.mkdir(os.path.join(path_to_unpack_folder, annotations_dir_name))
        if not os.path.exists(os.path.join(path_to_unpack_folder, images_dir_name)):
            os.mkdir(os.path.join(path_to_unpack_folder, images_dir_name))

        _num_jpegs_in_zips = 0
        file_names_in_folder = os.listdir(path_to_zips_folder)

        for name in file_names_in_folder:
            filename, file_extension = os.path.splitext(name)

            if file_extension == '.zip':
                print(name)

                # unpack .zip
                with zipfile.ZipFile(os.path.join(path_to_zips_folder, name), 'r') as zipObject:
                    listOfFileNames = zipObject.namelist()  # get filenames from zip
                    for member in listOfFileNames:
                        member_name = os.path.basename(member)

                        if member_name.startswith('.'):
                            continue

                        if member.endswith('.json') or member.endswith('.xml'):
                            target = open(os.path.join(
                                path_to_unpack_folder, annotations_dir_name, filename + '_' + member_name), "wb")
                        elif member.endswith('.jpg') or member.endswith('.jpeg') or member.endswith('.png'):
                            target = open(os.path.join(
                                path_to_unpack_folder, images_dir_name, member_name), "wb")
                            _num_jpegs_in_zips += 1
                        else:
                            continue

                        # copy a file from zip to a folder (annotations/images)
                        source = zipObject.open(member)
                        with source, target:
                            shutil.copyfileobj(source, target)

        print('Number of image files in all zips is {}'.format(_num_jpegs_in_zips))
        print()

        return os.path.join(path_to_unpack_folder, annotations_dir_name), \
               os.path.join(path_to_unpack_folder, images_dir_name)

    @staticmethod
    def unpack_from_folders(path_to_folders: str, path_to: str):
        """Сбор данных из распакованных папок

        Args:
            path_to_folders (str): Путь к распакованным папкам 
            path_to (str): Куда сохранять 

        Returns:
            (str, str): Путь к папке с json файлами | Путь к папке с картинками
        """
        print('...Unpack started...')

        annotations_dir_name = 'annotations'
        images_dir_name = 'images'

        # create separate dirs for annotations (.json) and images (.jpg)
        if not os.path.exists(os.path.join(path_to, annotations_dir_name)):
            os.mkdir(os.path.join(path_to, annotations_dir_name))
        if not os.path.exists(os.path.join(path_to, images_dir_name)):
            os.mkdir(os.path.join(path_to, images_dir_name))

        _num_jpegs_in_folders = 0
        dir_names_in_folder = os.listdir(path_to_folders)
        for dir_name in dir_names_in_folder:
            print(dir_name)

            for path, subdirs, files in os.walk(os.path.join(path_to_folders, dir_name)):
                for name in files:
                    file_extension = name.split('.')[-1]
                    file_name = ''.join(name.split('.')[:-1])

                    name = file_name + file_extension

                    if file_extension == 'json' or \
                            file_extension == 'xml':
                        target = os.path.join(
                            path_to, annotations_dir_name, dir_name + '_' + name)

                    elif file_extension == 'jpg' or \
                            file_extension == 'jpeg' or \
                            file_extension == 'png':
                        target = os.path.join(path_to, images_dir_name, name)

                        _num_jpegs_in_folders += 1

                    else:
                        continue

                    # copy a file from source to a folder (annotations/images)
                    source = os.path.join(path, name)

                    shutil.copyfile(source, target)

        print('Number of image files in all zips is {}'.format(_num_jpegs_in_folders))
        print()

        return os.path.join(path_to, annotations_dir_name), os.path.join(path_to, images_dir_name)

    @staticmethod
    def unite_jsons_from_list(jsons: list, categs: list = None, path_to_save: str = None, path_to_images: str = None):
        """Объединяет все coco json файлы в один 

        Args:
            jsons (list): массив из json.load объектов
            categs (list, optional): Категории объектов. Defaults to None.
            path_to_save (str, optional): Куда сохранить. Defaults to None.
            path_to_images (str, optional): Путь к картинкам. Defaults to None.

        Returns:
            json object: Объединенный json
        """
        print('...Uniting jsons...')

        united_json_data = JSON_HEADER

        if categs is None:
            categs = ["sliding",
                      "swing",
                      "roll-up"]

        categories = []
        for i, c in enumerate(categs):
            categories.append({
                "id": i + 1,
                "name": c,
                "supercategory": ""
            })

        category_name_to_id = {}
        for category in categories:
            category_name_to_id[category['name']] = category['id']

        united_json_data['categories'] = categories

        united_json_data['images'] = []
        united_json_data['annotations'] = []

        # variables to control that all ids are unique
        new_image_id = 1
        new_annotation_id = 1

        _images = []  # unique image names
        _nonexisting_images_num = 0

        for j, data in enumerate(jsons):
            _old_to_new_image_ids = {}
            _double_image_ids = []

            categories_for_this_json = data['categories']
            int_to_category_name = {}
            for category in categories_for_this_json:
                int_to_category_name[category['id']] = category['name']

            # copy unique images with new id
            for i, image in enumerate(data['images']):
                _old_to_new_image_ids[image['id']] = new_image_id
                image['id'] = new_image_id
                new_image_id += 1

                if image['file_name'] not in _images:
                    _images.append(image['file_name'])

                    if path_to_images is not None:
                        if not os.path.exists(os.path.join(path_to_images, image['file_name'])):
                            # print('file {} does not exist'.format(image['file_name']))
                            _nonexisting_images_num += 1

                    united_json_data['images'].append(image)

                else:
                    _double_image_ids.append(image['id'])

            for annotation in data['annotations']:
                new_image_id_ann = _old_to_new_image_ids[annotation['image_id']]

                if new_image_id_ann in _double_image_ids:
                    continue

                category_name = int_to_category_name[annotation['category_id']]
                if category_name not in category_name_to_id:
                    continue
                else:
                    annotation['category_id'] = category_name_to_id[category_name]

                annotation['image_id'] = new_image_id_ann

                annotation['id'] = new_annotation_id
                new_annotation_id += 1

                united_json_data['annotations'].append(annotation)
            new_image_id += 1

        print()
        print('Num of doubled images in annotations: {}'.format(
            len(_double_image_ids)))

        if path_to_images is not None:
            print('Num of non existing images in provided data: {}'.format(
                _nonexisting_images_num))
        else:
            print('Provide images path to get nonexisting images info')

        if path_to_save:
            with open(os.path.join(path_to_save, 'united_annotations.json'), 'w') as outfile:
                json.dump(united_json_data, outfile, indent=4)

        # CamerasDataParser.get_json_statistics(json_data=united_json_data)

        return united_json_data

    @staticmethod
    def unite_jsons_in_folder(path_to_annotations: str, path_to_images: str, cats: list,
                              save_filename: str = None, remove_files_after_uniting: bool = False):
        """Общий обработчик объединения json файлов в папке 
            - unite annotations in one .json | empty images are removed | cats is list of names of categories

        Args:
            path_to_annotations (str): Путь к папке с аннотациями 
            path_to_images (str): Путь к картинкам
            cats (list): Категории
            save_filename (str, optional): Название объединенного файла. Defaults to None.
            remove_files_after_uniting (bool, optional): Сохранять/удалять файлы после объединения. Defaults to False.

        Returns:
            json object: Объединенный json
        """
        print('...Uniting annotations...')

        # remove excisting output annotations not to get data from them
        if save_filename is not None:
            if os.path.exists(os.path.join(path_to_annotations, save_filename)):
                os.remove(os.path.join(path_to_annotations, save_filename))

        # get all annotations names
        annotation_names = os.listdir(path_to_annotations)
        jsons_list = []

        for j, filename in enumerate(annotation_names):
            print(filename)

            if filename.endswith('.xml'):
                data = CamerasDataParser.xml_to_json(os.path.join(path_to_annotations, filename))
            elif filename.endswith('.json'):
                with open(os.path.join(path_to_annotations, filename), 'r') as file:
                    data = json.load(file)
            else:
                print('Not annotations')
                continue

            jsons_list.append(data.copy())

            if remove_files_after_uniting:
                os.remove(os.path.join(path_to_annotations, filename))

        united_json_data = CamerasDataParser.unite_jsons_from_list(jsons_list, categs=cats,
                                                                   path_to_images=path_to_images)

        united_json_data = CamerasDataParser.check_and_remove_empty_images(
            united_json_data, path_to_images, path_to_removed=os.path.join(path_to_images, 'not_in_annotations'))

        if save_filename is not None:
            with open(os.path.join(path_to_annotations, save_filename), 'w') as outfile:
                json.dump(united_json_data, outfile, indent=4)

        return united_json_data

    @staticmethod
    def check_and_remove_empty_images(json_data: dict, path_to_images: str, path_to_removed: str = None):
        """Проверяет и удаляет картинки без аннотаций 

        Args:
            json_data (dict): массив из json.load объектов
            path_to_images (str): Путь к картинкам 
            path_to_removed (str, optional): Путь куда сохраняться картинки без аннотации. Defaults to None.

        Returns:
            json object: json_data
        """
        print('\n...Checking empty images...')

        print('Num of image files in final folder before deleting: {}'.format(
            len(os.listdir(path_to_images))))

        print('Num of images in json data before check: {}'.format(
            len(json_data['images'])))

        # -- make a dir for images without annotations --
        if path_to_removed is not None:
            if os.path.exists(path_to_removed):
                shutil.rmtree(path_to_removed)
            os.mkdir(path_to_removed)

        images_names_to_remove = []  # to remove from folder (names)
        images_to_remove = []  # to remove from json (objects)
        for image in json_data['images']:
            image_have_annotation = False
            for annotation in json_data['annotations']:
                if annotation['image_id'] == image['id']:
                    image_have_annotation = True
                    break
            if not image_have_annotation:
                images_names_to_remove.append(image['file_name'])
                images_to_remove.append(image)

        for im_name in images_names_to_remove:
            if os.path.exists(os.path.join(path_to_images, im_name)):
                if path_to_removed is not None:
                    shutil.copyfile(os.path.join(path_to_images, im_name), os.path.join(
                        path_to_removed, im_name))
                os.remove(os.path.join(path_to_images, im_name))
        for image in images_to_remove:
            json_data['images'].remove(image)

        print('Num of images in json data after check: {}'.format(
            len(json_data['images'])))

        print('Num of jpegs in final folder after deleting: {}'.format(
            len(os.listdir(path_to_images)) - 1))

        print()
        print('Num of removed images (without annotations): {}'.format(
            len(images_names_to_remove)))

        return json_data

    # save train, val and test data from classes to COCO format in given path
    @staticmethod
    def save_dataset_for_detector(path_to_annotations: str, path_to_save: str,
                                  train_percent: int = 88, val_percent: int = 5, treshold: int = 150):
        """Генерирует train/val/test jsons

        Args:
            path_to_annotations (str): Путь к аннотации
            path_to_save (str): Куда сохранить
            train_percent (int, optional): Процент тренировочной выборки. Defaults to 88.
            val_percent (int, optional): Процент валидационной выборки. Defaults to 5.
            treshold (int, optional): Если кол-во объектов меньше то этот класс не попадет в обучение. Defaults to 150.
        """
        print('Preparing data for detector...')

        with open(path_to_annotations, 'r') as file:
            json_data = json.load(file)

        data_statistics = CamerasDataParser.get_json_statistics(json_data=json_data)

        # choose the categories, which contain minimum num of annotations
        new_category_id = 1
        valid_categories = []
        valid_categories_ids = []
        for category in json_data['categories']:

            if data_statistics[category['name']] >= treshold:
                category['id'] = new_category_id
                valid_categories.append(category)
                valid_categories_ids.append(category['id'])
                new_category_id += 1

        images_with_annotations = []
        for image in json_data['images']:
            image_id = image['id']
            image['annotations'] = []

            for annotation in json_data['annotations']:
                if annotation['image_id'] == image_id and annotation['category_id'] in valid_categories_ids:
                    image['annotations'].append(annotation)

            if len(image['annotations']) == 0:
                continue
            else:
                images_with_annotations.append(image)

        # shuffled_data = shuffle(images_with_annotations)

        # dataset_size = len(shuffled_data)
        # train_dataset_size = int(dataset_size * train_percent / 100)
        # val_dataset_size = int(dataset_size * val_percent / 100)

        # train_dataset = shuffled_data[0:train_dataset_size]
        # val_dataset = shuffled_data[train_dataset_size:
        #                             train_dataset_size+val_dataset_size]
        # test_dataset = shuffled_data[train_dataset_size+val_dataset_size:]

        train_dataset, test_dataset = train_test_split(images_with_annotations, test_size=0.07, random_state=42)
        train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.07, random_state=42)

        print('Forming COCO data for detector...')

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        def form_and_save_dataset(images_with_annotations: list, path_to_save: str, categories: list):
            print('saving {}'.format(path_to_save))

            out_json = JSON_HEADER
            out_json['categories'] = categories
            out_json['images'] = []
            out_json['annotations'] = []

            for image in images_with_annotations:
                for annotation in image['annotations']:
                    out_json['annotations'].append(annotation)

                del image['annotations']

                out_json['images'].append(image)

            with open(path_to_save, 'w') as outfile:
                json.dump(out_json, outfile, indent=4)
                outfile.close()

        form_and_save_dataset(train_dataset, os.path.join(path_to_save, 'train.json'), valid_categories)
        form_and_save_dataset(val_dataset, os.path.join(path_to_save, 'val.json'), valid_categories)
        form_and_save_dataset(test_dataset, os.path.join(path_to_save, 'test.json'), valid_categories)

    @staticmethod
    def get_json_statistics(path_to_json: str = None, json_data: dict = None):
        """Статистика по json файлу

        Args:
            path_to_json (str, optional): Путь к json файлу (аннотации). Defaults to None.
            json_data (dict, optional): Или json.load объект. Defaults to None.

        Returns:
            dict: Статистика
        """
        if json_data is None:
            with open(path_to_json, 'r') as file:
                json_data = json.load(file)

        info_general = {
            'ALL_images': len(json_data['images']),
            'ALL_annotations': len(json_data['annotations'])
        }

        info_categories = {}

        categories = {}
        if 'categories' in json_data:
            for category in json_data['categories']:
                categories[category['id']] = category['name']

        for annotation in json_data['annotations']:
            category = categories[annotation['category_id']]

            if category in info_categories:
                info_categories[category] += 1
            else:
                info_categories[category] = 1

        for key, value in info_general.items():
            print("{0}: ({1})".format(key, value))
        print()

        for k, v in sorted(info_categories.items(), key=lambda item: item[1], reverse=True):
            print("{0}: ({1})".format(k, v))
        print()

        info_general.update(info_categories)

        return info_general

    @staticmethod
    def show_annotations(json_data: dict, images_folder_path: str, output_folder: str, num_to_show: int = 10):
        """Сохраняет картики с визуальной разметкой

        Args:
            json_data (dict): Аннотация
            images_folder_path (str): Путь к картинкам
            output_folder (str): Путь куда сохранять
            num_to_show (int, optional): Кол-во. Defaults to 10.

        """
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        def draw_text(frame, text, box_corner, index, shift):
            position = (int(box_corner[0]) + 5, int(box_corner[1]) + index * shift)

            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        def draw_predictions(frame, annotations, categories, im_height: int = 2448, im_width: int = 2048):
            v = Visualizer(frame, instance_mode=ColorMode.IMAGE_BW)

            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                      'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

            for k, a in enumerate(annotations):
                box = [a['bbox'][0], a['bbox'][1], a['bbox'][2] + a['bbox'][0], a['bbox'][3] + a['bbox'][1]]
                mask = GenericMask([a['segmentation']], height=im_height, width=im_width).mask

                v.draw_box(box, edge_color=colors[k % len(colors)])
                v.draw_binary_mask(mask, color=colors[k % len(colors)])

            img = v.get_output().get_image()
            img = np.hstack((img, frame))

            for annotation in annotations:
                shift = 30
                text = categories[annotation['category_id']]
                draw_text(img, text, annotation['bbox'][:2], 1, shift)

            return img

        if num_to_show == -1:
            num_to_show = len(json_data['images'])

        categories = {}
        for category in json_data['categories']:
            categories[category['id']] = category['name']

        image_id_to_name = {}
        images_with_annotations = {}
        for image in json_data['images']:
            images_with_annotations[image['file_name']] = []
            image_id_to_name[image['id']] = image['file_name']

        for annotation in json_data['annotations']:
            images_with_annotations[image_id_to_name[annotation['image_id']]].append(annotation)

        for i, image_name in enumerate(images_with_annotations):
            if i >= num_to_show:
                break

            image_full_path = os.path.join(images_folder_path, image_name)

            im = cv2.imread(image_full_path)[:, :, ::-1]
            im_gt = draw_predictions(im, images_with_annotations[image_name], categories)

            cv2.imwrite(os.path.join(output_folder, image_name), im_gt[:, :, ::-1])

            print('\r[{}/{}]'.format(i + 1, num_to_show), end='')

    @staticmethod
    def xml_to_json(path_from: str, output_file_name: str = None):
        """XML 2 JSON
        """
        tree = ElementTree.parse(path_from)
        root = tree.getroot()  # <annotations>

        # --- define categories
        categories = []
        category_name_to_id = {}
        cat_id = 1

        labels = root.find('meta').find('task').find('labels')
        for label in labels:
            label_name = label.find('name').text

            categories.append({'id': cat_id,
                               'name': label_name,
                               'supercategory': ''})
            category_name_to_id[label_name] = cat_id

            cat_id += 1
        # ---

        output_json = JSON_HEADER

        output_json['categories'] = categories
        output_json['images'] = []
        output_json['annotations'] = []

        image_id = 1
        annotation_id = 1

        for image in root.iter('image'):
            image_name = image.get('name')
            image_width = image.get('width')
            image_height = image.get('height')

            output_json['images'].append({'id': image_id,
                                          'width': int(image_width),
                                          'height': int(image_height),
                                          'file_name': str(image_name),
                                          'license': 0,
                                          'flickr_url': "",
                                          'coco_url': "",
                                          'date_captured': 0})

            for polygon in image.iter('polygon'):
                category_name = polygon.get('label')
                category_id = category_name_to_id[category_name]

                segmentation = list(map(float, re.split(',|;', polygon.get('points'))))
                contour = np.expand_dims(np.array(segmentation, dtype=np.int32).reshape((-1, 2)), axis=1)
                area = cv2.contourArea(contour)

                x1 = min(contour[:, :, 0])[0]
                y1 = min(contour[:, :, 1])[0]
                x2 = max(contour[:, :, 0])[0]
                y2 = max(contour[:, :, 1])[0]
                w = x2 - x1
                h = y2 - y1
                bbox = [float(x1), float(y1), float(w), float(h)]

                output_json['annotations'].append({'id': annotation_id,
                                                   'image_id': image_id,
                                                   'category_id': category_id,
                                                   'segmentation': [segmentation],
                                                   'area': area,
                                                   'bbox': bbox,
                                                   'iscrowd': 0,
                                                   'attributes': {"occluded": False}
                                                   })

                annotation_id += 1

            image_id += 1

        if output_file_name is not None:
            with open(output_file_name, 'w') as outfile:
                json.dump(output_json, outfile, indent=4)

        return output_json

    @staticmethod
    def make_images_folder_from_json(path_to_all_images: str, path_to_save: str, path_to_json: str):
        """Создает папки с картинками из json файлов (Юзайте для <train/test/val>.json)

        Args:
            path_to_all_images (str): Все картинки
            path_to_save (str): куда сохранять 
            path_to_json (str): путь к json файлу
        """
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        with open(path_to_json, 'r') as file:
            gt_data = json.load(file)

        for i, image in enumerate(gt_data['images']):
            print('\r{}/{}'.format(i, len(gt_data['images'])), end='')

            image_name = image['file_name']

            src = os.path.join(path_to_all_images, image_name)
            dst = os.path.join(path_to_save, image_name)

            shutil.copy(src, dst)


def main():
    """USAGE EXAMPLES
    """

    # # -- Use to unzip and unite data from annotators --
    # path_from = 'project/zips/data_labeled'
    # path_to = 'project/datasets/data'

    # CamerasDataParser.unpack_and_unite_zips(path_to_zips_folder=path_from, path_to_unpack_folder=path_to)
    # CamerasDataParser.get_json_statistics(path_to_json='/raid/nanosemantics/CV/digitalRoads/gates_doors/datasets/gates_131022/annotations/united_annotations.json')


    # # -- Use to form dataset --
    # path_to_annotations='project/annotations/united_annotations.json'
    # path_to_save='project/datasets/data/split'

    # CamerasDataParser.save_dataset_for_detector(path_to_annotations=path_to_annotations,
    #                                             path_to_save=path_to_save)


    # # -- Check stats --
    # CamerasDataParser.get_json_statistics(path_to_json='project/datasets/data/split/train.json')
    # CamerasDataParser.get_json_statistics(path_to_json='project/datasets/data/split/val.json')
    # CamerasDataParser.get_json_statistics(path_to_json='project/datasets/data/split/test.json')


    # -- Use to visualize annotations --
    # path_to_json = 'project/datasets/data/split/test.json'
    # in_path = 'project/datasets/data/images'
    # out_path = 'project/datasets/data/check_test_data'

    # with open(path_to_json, 'r') as file:
    #     json_data = json.load(file)

    # CamerasDataParser.show_annotations(json_data=json_data, images_folder_path=in_path,
    #                                    output_folder=out_path, num_to_show=-1)


    # #-- Use to make input folder for script testing
    # path_to_test_json = 'project/datasets/data/split/test.json'
    # path_to_images = 'project/datasets/data/split/images'
    # path_to_save = 'project/datasets/data/split/test_images'

    # CamerasDataParser.make_images_folder_from_json(path_to_all_images=path_to_images,
    #                                                path_to_save=path_to_save,
    #                                                path_to_json=path_to_test_json)


if __name__ == '__main__':
    main()
