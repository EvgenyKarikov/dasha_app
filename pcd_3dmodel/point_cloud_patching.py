import numpy as np
import open3d as o3d

def estimate_patch_normal(patch, k_neighbors=30):
    """
    Вычисляет среднюю нормаль для патча с использованием Open3D.
    
    Args:
        patch (np.ndarray): Патч облака точек размером (M, 3).
        k_neighbors (int): Количество соседей для оценки нормалей.
    
    Returns:
        np.ndarray: Средняя нормаль патча (вектор размером (3,)).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(patch)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    normals = np.asarray(pcd.normals)
    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)  # Нормализуем
    if mean_normal[2] < 0:  # Ориентируем нормаль вверх (z > 0)
        mean_normal = -mean_normal
    return mean_normal

def rotate_to_align_with_z(normal):
    """
    Вычисляет матрицу поворота для выравнивания нормали с осью z.
    
    Args:
        normal (np.ndarray): Вектор нормали (3,).
    
    Returns:
        np.ndarray: Матрица поворота 3x3.
    """
    z_axis = np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)
    if np.allclose(normal, z_axis):
        return np.eye(3)
    
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, z_axis)
    
    if s == 0:
        return np.eye(3) if c > 0 else np.diag([1, 1, -1])
    
    v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + v_skew + v_skew @ v_skew * ((1 - c) / (s ** 2))
    return rotation_matrix

def create_patches(point_cloud, patch_size=10000, overlap_ratio=0.1):
    """
    Делит облако точек на патчи заданного размера с перекрытием, выполняет нормализацию,
    переориентацию и усиление глубины.
    
    Args:
        point_cloud (np.ndarray): Облако точек размером (N, 3).
        patch_size (int): Количество точек в одном патче.
        overlap_ratio (float): Доля перекрытия между соседними патчами (0.0–1.0).
    
    Returns:
        list: Список патчей (каждый — np.ndarray размером (patch_size, 3)).
    """
    num_points = point_cloud.shape[0]
    if num_points < patch_size:
        raise ValueError("Облако точек слишком маленькое для заданного размера патча.")
    
    patches = []
    step_size = int(patch_size * (1 - overlap_ratio))  # Размер шага с учетом перекрытия
    
    # Разделяем облако точек на патчи
    for start_idx in range(0, num_points, step_size):
        end_idx = min(start_idx + patch_size, num_points)
        patch = point_cloud[start_idx:end_idx].copy()
        
        # Пропускаем патчи, которые меньше заданного размера
        if patch.shape[0] < patch_size:
            continue
        
        # Нормализация: центроид патча в (0, 0, 0)
        centroid = np.mean(patch, axis=0)
        patch -= centroid
        
        # Переориентация: выравнивание средней нормали с осью z
        mean_normal = estimate_patch_normal(patch)
        rotation_matrix = rotate_to_align_with_z(mean_normal)
        patch = patch @ rotation_matrix.T  # Поворачиваем патч
        
        # Усиление глубины: умножаем z-координаты на коэффициент alpha
        alpha = 10.0  # Как указано в статье
        patch[:, 2] *= alpha
        
        patches.append(patch)
    
    return patches

# Пример использования
if __name__ == "__main__":
    # Создаем синтетическое облако точек для демонстрации
    num_points = 50000
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    z = np.sin(np.sqrt(x**2 + y**2)) * 0.1  # Имитация поверхности
    point_cloud = np.vstack((x, y, z)).T
    
    # Параметры
    patch_size = 10000  # Количество точек в патче
    overlap_ratio = 0.1  # 10% перекрытия
    
    # Делим облако точек на патчи
    patches = create_patches(point_cloud, patch_size, overlap_ratio)
    
    # Сохраняем патчи для визуализации (опционально)
    for i, patch in enumerate(patches):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(patch)
        o3d.io.write_point_cloud(f"patch_{i}.ply", pcd)
        print(f"Сохранен патч {i} с {patch.shape[0]} точками")
    
    print(f"Создано {len(patches)} патчей")