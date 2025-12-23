import os,sys
import time
import multiprocessing as mp
from multiprocessing import Queue, Process
import asyncio

from ultralytics import YOLO

import yaml
import json
from rich import print


curdir = os.path.dirname(__file__)
conf_path = os.path.join(curdir,'conf.yaml')

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

async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info('peername')
    print(f"{addr} Connected")
    
    # Уникальный ID задачи
    task_id = int(time.time() * 1000)
    
    # Создаём временную очередь для ответа конкретно этой задачи
    reply_queue: Queue = mp.Queue()
    
    try:
        # Читаем данные (изображение) от клиента
        data = await reader.read(10 * 1024 * 1024)  # до 10 МБ
        if not data:
            return
        
        # Кладём задачу в общую очередь воркеров
        task_queue.put((task_id, data, reply_queue))
        
        # Ждём результат от воркера (с таймаутом)
        result_data = reply_queue.get(timeout=30)  # 30 сек таймаут
        
        # Отправляем JSON обратно клиенту
        response = json.dumps(result_data, ensure_ascii=False).encode('utf-8')
        writer.write(response + b'\n')
        await writer.drain()
        
    except Exception as e:
        error_resp = json.dumps({"task_id": task_id, "success": False, "error": str(e)}).encode()
        writer.write(error_resp + b'\n')
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()
        print(f"Соединение с {addr} закрыто")

# ------------------- Запуск сервера и воркеров -------------------
def start_workers(num_workers: int = 4, model_path: str = "yolov8n.pt"):
    global task_queue
    task_queue = Queue()
    
    workers = []
    for _ in range(num_workers):
        p = Process(target=yolo_worker, args=(task_queue, model_path), daemon=True)
        p.start()
        workers.append(p)
    
    return workers, task_queue

async def main(host: str = '0.0.0.0', 
               num_workers: int = 4, 
               conf: dict={}):
    
    port = conf['conf']['command_listen_port']
    yolo_weights_path = conf['conf']['yolo_weights_path']
    
    workers, _ = start_workers(num_workers=num_workers,
                               model_path=yolo_weights_path)
    
    server = await asyncio.start_server(handle_client, host, port)
    
    print(f"application listen at {host}:{port}")
    print(f"run {num_workers} YOLO-workers")
    
    async with server:
        await server.serve_forever()
    
    # Graceful shutdown (если нужно)
    # for _ in workers:
    #     task_queue.put((None, None, None))

if __name__=='__main__':        
    with open(conf_path) as f:    
        conf = yaml.safe_load(f)        
        
    mp.set_start_method('spawn')  # 'fork' может вызывать проблемы с CUDA
    
    try:
        asyncio.run(main(num_workers=4,conf=conf))
    except KeyboardInterrupt:
        print("application stopped")    
    pass