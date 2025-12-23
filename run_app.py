import os,sys
import time
import multiprocessing as mp
from multiprocessing import Queue, Process
import asyncio

import yaml
from rich import print


curdir = os.path.dirname(__file__)
conf_path = os.path.join(curdir,'conf.yaml')

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

async def main(host: str = '0.0.0.0', port: int = 8888, num_workers: int = 4):
    workers, _ = start_workers(num_workers=num_workers)
    
    server = await asyncio.start_server(handle_client, host, port)
    
    print(f"TCP-сервер запущен на {host}:{port}")
    print(f"Запущено {num_workers} YOLO-воркеров")
    
    async with server:
        await server.serve_forever()
    
    # Graceful shutdown (если нужно)
    # for _ in workers:
    #     task_queue.put((None, None, None))

if __name__=='__main__':
    with open(conf_path) as f:    
        conf = yaml.safe_load(f)
    pass