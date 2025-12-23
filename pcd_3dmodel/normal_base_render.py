import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def depthmap2normalmap(depth_map) -> None:
        """Converts the depth map image to a normal map image.

        Args:
            output_path (str): The path to save the normal map image file.

        """
        rows, cols = depth_map.shape

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Calculate the partial derivatives of depth with respect to x and y
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1)

        # Compute the normal vector for each pixel
        normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal = (normal + 1) * 127.5
        normal = normal.clip(0, 255).astype(np.uint8)

        # Save the normal map to a file
        normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        return normal_bgr

def estimate_normals(point_cloud, k_neighbors=30):
    """
    Вычисляет нормали для облака точек с использованием Kd-tree и метода наименьших квадратов.
    
    Args:
        point_cloud (np.ndarray): Облако точек размером (N, 3) с координатами (x, y, z).
        k_neighbors (int): Количество соседей для оценки нормали.
    
    Returns:
        np.ndarray: Массив нормалей размером (N, 3) для каждой точки.
    """
    # Создаем объект PointCloud для Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Оцениваем нормали с использованием Kd-tree
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
    )
    
    # Извлекаем нормали
    normals = np.asarray(pcd.normals)
    
    # Переориентируем нормали в направлении оси z (виртуальная камера)
    # Предполагаем, что нормали должны быть направлены "наружу" (z > 0)
    for i in range(len(normals)):
        if normals[i, 2] < 0:  # Если нормаль направлена вниз, инвертируем
            normals[i] = -normals[i]
    
    return normals

def project_to_depth_image_o3d(point_cloud, resolution=(256, 256)):
    """
    Проецирует облако точек в 2D-глубинную карту с использованием Open3D.
    
    Args:
        point_cloud (np.ndarray): Облако точек размером (N, 3).
        resolution (tuple): Разрешение выходного изображения (height, width).
    
    Returns:
        np.ndarray: Глубинная карта размером (height, width), нормализованная в [0, 255], тип uint8.
    """
    height, width = resolution
    
    # Создаем PointCloud для Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    
    # Определяем параметры камеры
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=width / 2.0,  # Фокусное расстояние по x (примерное значение)
        fy=height / 2.0,  # Фокусное расстояние по y
        cx=width / 2.0,   # Центр проекции по x
        cy=height / 2.0   # Центр проекции по y
    )
    
    # Устанавливаем внешние параметры камеры (смотрим вдоль оси z)
    extrinsic = np.eye(4)
    extrinsic[2, 3] = -np.max(np.abs(point_cloud[:, 2])) * 2.0  # Смещение камеры по z
    
    # Проецируем точки в 2D-координаты
    points_3d = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))  # Гомогенные координаты
    projected_points = (intrinsic.intrinsic_matrix @ np.linalg.inv(extrinsic)[:3, :] @ points_3d.T).T
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # Перспективная проекция
    u = np.clip(projected_points[:, 0], 0, width - 1).astype(int)
    v = np.clip(projected_points[:, 1], 0, height - 1).astype(int)
    
    # Создаем глубинную карту
    depth_image = np.zeros((height, width), dtype=np.float32)
    for i, (ui, vi) in enumerate(zip(u, v)):
        z = point_cloud[i, 2]
        if depth_image[vi, ui] == 0 or z < depth_image[vi, ui]:  # Ближайшая глубина
            depth_image[vi, ui] = z
    
    # Заполняем пропуски (нулевые значения) с помощью интерполяции
    mask = depth_image == 0
    if np.any(mask):
        grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
        points = np.column_stack((point_cloud[:, 0], point_cloud[:, 1]))
        values = point_cloud[:, 2]
        interpolated = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.mean(values))
        depth_image[mask] = interpolated[mask]
    
    # Сглаживание для устранения шума
    depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
    
    # Нормализация в диапазон [0, 255]
    depth_min, depth_max = depth_image.min(), depth_image.max()
    if depth_max > depth_min:
        depth_image = (depth_image - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_image = np.zeros_like(depth_image)
    
    return depth_image.astype(np.uint8)


def project_to_depth_image(point_cloud, resolution=(256, 256)):
    """
    Проецирует облако точек в 2D-глубинную карту.
    
    Args:
        point_cloud (np.ndarray): Облако точек размером (N, 3).
        resolution (tuple): Разрешение выходного изображения (height, width).
    
    Returns:
        np.ndarray: Глубинная карта размером (height, width).
    """
    height, width = resolution
    depth_image = np.zeros((height, width), dtype=np.float32)
    
    # Нормализуем координаты x, y в диапазон [0, 1] для проекции
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    if x_max == x_min or y_max == y_min:
        raise ValueError("Облако точек слишком мало или вырождено.")
    
    # Преобразуем x, y в пиксельные координаты
    x_normalized = (x - x_min) / (x_max - x_min) * (width - 1)
    y_normalized = (y - y_min) / (y_max - y_min) * (height - 1)
    
    # Округляем до ближайших пикселей
    x_idx = np.clip(x_normalized.astype(int), 0, width - 1)
    y_idx = np.clip(y_normalized.astype(int), 0, height - 1)
    
    # Заполняем глубинную карту значениями z
    for i in range(len(point_cloud)):
        depth_image[y_idx[i], x_idx[i]] = z[i]
    
    
    # Создаем маску для пропусков (нулевых значений)
    mask = (depth_image == 0).astype(np.uint8) * 255
    
    # # Интерполяция пропусков с использованием OpenCV inpaint
    # if np.any(mask):
    #     depth_image = cv2.inpaint(
    #         depth_image.astype(np.float32),
    #         mask,
    #         inpaintRadius=0.01,  # Радиус интерполяции (можно настроить)
    #         flags=cv2.INPAINT_NS  # Navier-Stokes метод для гладкой интерполяции
    #     )
    
    # Заполняем пропуски (опционально, если нужно)
    depth_image[depth_image == 0] = np.mean(z)  # Заполняем нули средним значением z
    
    # Нормализуем глубину для визуализации (в диапазон [0, 255])
    depth_min, depth_max = depth_image.min(), depth_image.max()
    if depth_max > depth_min:
        depth_image = (depth_image - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_image = np.zeros_like(depth_image)
    
    return depth_image

def project_to_depth_image_interpolated(point_cloud, resolution=(256, 256)):
    """
    Проецирует облако точек в 2D-глубинную карту с заполнением пропусков.
    
    Args:
        point_cloud (np.ndarray): Облако точек размером (N, 3).
        resolution (tuple): Разрешение выходного изображения (height, width).
    
    Returns:
        np.ndarray: Глубинная карта размером (height, width).
    """
    height, width = resolution
    depth_image = np.zeros((height, width), dtype=np.float32)
    
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    if x_max == x_min or y_max == y_min:
        raise ValueError("Облако точек слишком мало или вырождено.")
    
    # Нормализуем координаты x, y в диапазон [0, width-1] и [0, height-1]
    x_normalized = (x - x_min) / (x_max - x_min) * (width - 1)
    y_normalized = (y - y_min) / (y_max - y_min) * (height - 1)
    
    # Создаем сетку для интерполяции
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.vstack((x_normalized, y_normalized)).T
    values = z
    
    # Интерполяция для заполнения пропусков
    depth_image = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.mean(z))
    
    # Заполняем оставшиеся NaN средним значением
    depth_image = np.nan_to_num(depth_image, nan=np.mean(z))
    
    # Сглаживание для устранения шума
    # depth_image = cv2.GaussianBlur(depth_image, (5, 5), 0)
    
    # Нормализуем для визуализации (0–255)
    depth_min, depth_max = depth_image.min(), depth_image.max()
    if depth_max > depth_min:
        depth_image = (depth_image - depth_min) / (depth_max - depth_min) * 255.0
    else:
        depth_image = np.zeros_like(depth_image)
    
    return depth_image

def project_to_rendered_image(normals, point_cloud, resolution=(256, 256)):
    """
    Проецирует нормали в рендерированное изображение (RGB).
    
    Args:
        normals (np.ndarray): Нормали размером (N, 3) с компонентами (nx, ny, nz).
        point_cloud (np.ndarray): Облако точек размером (N, 3) для координат.
        resolution (tuple): Разрешение выходного изображения (height, width).
    
    Returns:
        np.ndarray: Рендерированное изображение размером (height, width, 3), тип uint8.
    """
    height, width = resolution
    rendered_image = np.zeros((height, width, 3), dtype=np.float32)
    
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    if x_max == x_min or y_max == y_min:
        raise ValueError("Облако точек слишком мало или вырождено.")
    
    x_normalized = (x - x_min) / (x_max - x_min) * (width - 1)
    y_normalized = (y - y_min) / (y_max - y_min) * (height - 1)
    
    normals_normalized = (normals + 1.0) / 2.0
    
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    points = np.vstack((x_normalized, y_normalized)).T
    
    for channel in range(3):
        values = normals_normalized[:, channel]
        channel_image = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.mean(values))
        channel_image = np.nan_to_num(channel_image, nan=np.mean(values))
        channel_image = cv2.GaussianBlur(channel_image, (5, 5), 0)
        rendered_image[:, :, channel] = channel_image * 255.0
    
    return rendered_image.astype(np.uint8)

def normal_based_rendering(point_cloud, resolution=(256, 256), k_neighbors=30, combine_method='multichannel'):
    """
    Выполняет нормаль-базированный рендеринг: вычисляет нормали и проецирует в 2D.
    
    Args:
        point_cloud (np.ndarray): Облако точек размером (N, 3).
        resolution (tuple): Разрешение выходных изображений (height, width).
        k_neighbors (int): Количество соседей для оценки нормалей.
        combine_method (str): Метод объединения ('multichannel', 'horizontal', 'vertical', 'none').
    
    Returns:
        np.ndarray or tuple: Объединенное изображение или (depth_image, rendered_image).
    """
    normals = estimate_normals(point_cloud, k_neighbors)
    depth_image = project_to_depth_image(point_cloud, resolution)
    
    
    if combine_method == 'multichannel':
        rendered_image = project_to_rendered_image(normals, point_cloud, resolution)
        # Объединяем в 4-канальное изображение (RGB + глубина)
        combined_image = np.zeros((resolution[0], resolution[1], 4), dtype=np.uint8)
        combined_image[:, :, :3] = rendered_image
        combined_image[:, :, 3] = depth_image
    elif combine_method == 'horizontal':
        # Конкатенация по горизонтали (глубина преобразована в RGB)
        rendered_image = project_to_rendered_image(normals, point_cloud, resolution)
        depth_rgb = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
        combined_image = np.hstack((rendered_image, depth_rgb))
    elif combine_method == 'vertical':
        # Конкатенация по вертикали
        rendered_image = project_to_rendered_image(normals, point_cloud, resolution)
        depth_rgb = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)
        combined_image = np.vstack((rendered_image, depth_rgb))
    else:  # 'none'
        rendered_image = np.zeros_like(depth_image)
        combined_image = np.zeros_like(depth_image)
    return depth_image, rendered_image, combined_image

# Пример использования
if __name__ == "__main__":
    # Создаем синтетическое облако точек для демонстрации
    num_points = 10000
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    z = np.sin(np.sqrt(x**2 + y**2)) * 0.1  # Имитация поверхности с небольшими дефектами
    point_cloud = np.vstack((x, y, z)).T
    
    # Выполняем нормаль-базированный рендеринг
    depth_image, rendered_image = normal_based_rendering(point_cloud, resolution=(256, 256), k_neighbors=30)
    
    # Сохраняем изображения для визуализации
    cv2.imwrite("depth_image.png", depth_image.astype(np.uint8))
    cv2.imwrite("rendered_image.png", rendered_image.astype(np.uint8))
    
    print("Глубинная карта и рендерированное изображение сохранены как depth_image.png и rendered_image.png")