import os
import sys
import socket
import struct
import msgpack
from collections import deque
import time

import numpy as np

import multiprocessing as mp
from multiprocessing import Queue, Process
import asyncio

from ultralytics import YOLO

import yaml
import json
from rich import print


curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, 'profitalk')))
sys.path.append(os.path.abspath(os.path.join(curdir, 'pcd_3dmodel')))

from pcd_3dmodel.normal_base_render import (normal_based_rendering,
                                            project_to_depth_image,
                                            estimate_normals,
                                            project_to_rendered_image,
                                            depthmap2normalmap)
from pcd_3dmodel.point_cloud_patching import create_patches
from profitalk.profitalk import ProfiTalkClient

conf_path = os.path.join(curdir, 'conf.yaml')



# ------------------- Воркер-функция -------------------
def yolo_worker(task_queue: Queue, model_path: str = "yolov8n.pt"):
    # Загружаем модель внутри процесса (важно!)
    model = YOLO(model_path)

    while True:
        # Получаем задачу: (task_id, image_bytes, reply_queue)
        task_id, image_bytes, reply_queue = task_queue.get()

        if task_id is None:  # Сигнал завершения
            break

        try:
            # Выполняем инференс
            results = model(image_bytes, verbose=False)
            # Формируем результат (можно упростить/расширить)
            result_data = {
                "task_id": task_id,
                "success": True,
                "detections": [
                    {
                        "class": r.names[int(r.boxes.cls[0])],
                        "confidence": float(r.boxes.conf[0]),
                        "box": r.boxes.xyxy[0].tolist()
                    }
                    for r in results if r.boxes is not None
                ]
            }
        except Exception as e:
            result_data = {
                "task_id": task_id,
                "success": False,
                "error": str(e)
            }

        # Отправляем результат обратно клиенту через reply_queue
        reply_queue.put(result_data)


def rf627_worker(scanner: ProfiTalkClient,dimensions: tuple, profile_buffer: Queue,task_queue: Queue, conf: dict):        
            
    while True:
        
        # collect profiles from scanner
        pr = profile_buffer.get()
        print(f'received: {len(pr)}')
        params = scanner.read_parameters()
        divider = params['user_trigger_sync_divider']
        xmin,xmax,zmin = dimensions
        # Получаем задачу: (task_id, image_bytes, reply_queue)
        
        try:
            task_id, task_msg, reply_queue = task_queue.get(timeout=1.0)
            print(f'received id:{task_id}, msg:{task_msg}')
            wheel_d = task_msg['wheel_diameter']
            wheel_width = task_msg['wheel_width']
            wheel_height = task_msg['wheel_height']

            # compute geometric parameters
            w_radius_m = (0.5*wheel_d*25.4 + wheel_width *
                          wheel_height/100)*1e-3  # meters
            pulses_per_revolute = 429687.5
            measures_per_revolute = pulses_per_revolute/divider
            radians_per_pulse = 2*np.pi/pulses_per_revolute
            radians_per_measure = 2*np.pi/measures_per_revolute

            meter_per_pulse = w_radius_m*radians_per_pulse  # meters
            meter_per_measure = w_radius_m*radians_per_measure

            scan_width_m = float(xmax-xmin)*1e-3  # meters

            num_ms = 1000*scan_width_m/1456
            num_pulses = int(num_ms/meter_per_measure)
            print(f'pulses to frame {num_pulses}')
            pass
        except:
            pass
        pass

        # if task_id is None:  # Сигнал завершения
        #     break

        # wheel_d = task_msg['wheel_diameter']
        # wheel_width = task_msg['wheel_width']
        # wheel_height = task_msg['wheel_height']

        # compute geometric parameters
        # w_radius_m = (0.5*wheel_d*25.4 + wheel_width *
        #               wheel_height/100)*1e-3  # meters
        # pulses_per_revolute = 429687.5
        # measures_per_revolute = pulses_per_revolute/divider
        # radians_per_pulse = 2*np.pi/pulses_per_revolute
        # radians_per_measure = 2*np.pi/measures_per_revolute

        # meter_per_pulse = w_radius_m*radians_per_pulse  # meters
        # meter_per_measure = w_radius_m*radians_per_measure

        # scan_width_m = float(xmax-xmin)*1e-3  # meters

        # num_ms = 1000*scan_width_m/1456
        # num_pulses = int(num_ms/meter_per_measure)
        

        # try:

        #     result_data = {
        #         "task_id": task_id,
        #         "success": True,
        #         "detections": [

        #         ]
        #     }
        # except Exception as e:
        #     result_data = {
        #         "task_id": task_id,
        #         "success": False,
        #         "error": str(e)
        #     }

        # # Отправляем результат обратно клиенту через reply_queue
        # reply_queue.put(result_data)

        pass

    pass


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    print(f"{addr} Connected")

    # Уникальный ID задачи
    task_id = int(time.time() * 1000)

    # Создаём временную очередь для ответа конкретно этой задачи
    

    try:
        # Читаем данные (изображение) от клиента
        data = await reader.read(10 * 1024 * 1024)  # до 10 МБ
        if not data:
            return

        msg = data.decode()

        if msg == 'req':

            # this parameters should be received in msg
            wheel_d = 17  # r17
            wheel_width = 235
            wheel_height = 65

            print(f'received request from {addr}')

            task_msg = {
                'command': 'start',
                'wheel_diameter': wheel_d,
                'wheel_height': wheel_height,
                'wheel_width': wheel_width
            }

            profiler_task_queue.put((task_id, task_msg, reply_queue))

        # # Кладём задачу в общую очередь воркеров
        # task_queue.put((task_id, data, reply_queue))

        # # Ждём результат от воркера (с таймаутом)
        # result_data = reply_queue.get(timeout=30)  # 30 сек таймаут

        # # Отправляем JSON обратно клиенту
        # response = json.dumps(result_data, ensure_ascii=False).encode('utf-8')
        # writer.write(response + b'\n')
        # await writer.drain()

    except Exception as e:
        error_resp = json.dumps(
            {"task_id": task_id, "success": False, "error": str(e)}).encode()
        writer.write(error_resp + b'\n')
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Соединение с {addr} закрыто")

# ------------------- Запуск сервера и воркеров -------------------


def start_yolo_workers(num_workers: int = 4, model_path: str = "yolov8n.pt"):
    

    workers = []
    for _ in range(num_workers):
        p = Process(target=yolo_worker, args=(
            yolo_task_queue, model_path), daemon=True)
        p.start()
        workers.append(p)

    return workers, yolo_task_queue


def start_profiler(profiler_type: str = '', profiler_conf: dict = {}):
    def init_dimensions():
        stream = scanner.collect_profiles_stream(50)
        points = []
        intensity = []
        for p in stream:
            pnts, intens = ProfiTalkClient.process_single_profile(p)
            points+=pnts
            intensity+=intens
            
        p = np.array(points)
        xmin = np.min(p[:,0])
        xmax = np.max(p[:,0])
        zmin = np.min(p[:,2])
        return xmin, xmax, zmin
        pass
           
    if profiler_type == 'rf627':
        
        scanners = ProfiTalkClient.discover_scanners()

        ip = profiler_conf['addr']

        scanner = ProfiTalkClient(ip)
        params = scanner.read_parameters()

        print(f'connect to rf627 scanner with {ip} address')

        divider = params['user_trigger_sync_divider']
        # read all parameters descriptions from scanner
        descr = scanner.read_parameters_description()
        xmin,xmax,zmin = init_dimensions()
        print('geometries init')
        
        # scanner.get_profiles_stream(q=profiles_buffer)
        receiver = Process(target=scanner._stream_reader,
                           args=(scanner.profiles_port,
                                 profiles_buffer),
                           daemon=True)        
        receiver.start()        
                
        worker = Process(target=rf627_worker, args=(scanner,
                                                    (xmin,xmax,zmin),
                                                    profiles_buffer,
                                                    profiler_task_queue,
                                                    profiler_conf), daemon=True)
        worker.start()

    return worker, profiler_task_queue
    pass


async def main(host: str = '0.0.0.0',
               num_workers: int = 4,
               conf: dict = {}):

    port = conf['conf']['command_listen_port']
    yolo_weights_path = conf['conf']['yolo_weights_path']

    profiler_type = conf['conf']['profiler_type']
    if profiler_type == 'rf627':
        profiler_conf = conf['conf']['rf627']            

    profiler_worker = start_profiler(profiler_type, profiler_conf)

    workers, _ = start_yolo_workers(num_workers=num_workers,
                                    model_path=yolo_weights_path)
    server = await asyncio.start_server(handle_client, host, port)

    print(f"application listen at {host}:{port}")
    print(f"run {num_workers} YOLO-workers")

    async with server:
        await server.serve_forever()

    # Graceful shutdown (если нужно)
    # for _ in workers:
    #     task_queue.put((None, None, None))

if __name__ == '__main__':
    
    test_msg = '{"Par":{"Width":+265.000,"Height":+55.000,"Radius":+14.000},"Pos":{"Xmz":+103.412,"Ymz":+93.502,"Y":+878.371},"State":0}'
    test_msg_json=json.loads(test_msg.replace('+',''))
    
    with open(conf_path) as f:
        conf = yaml.safe_load(f)

    mp.set_start_method('spawn')  # 'fork' может вызывать проблемы с CUDA
    with mp.Manager() as manager:
        profiles_buffer = manager.Queue()
        profiler_task_queue = manager.Queue()
        yolo_task_queue = manager.Queue()
        reply_queue = manager.Queue()
        try:
            asyncio.run(main(num_workers=4, conf=conf))
        except KeyboardInterrupt:
            print("application stopped")
    pass
