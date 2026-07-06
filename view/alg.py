import numpy as np
import open3d as o3d
from config import *
from scipy.spatial import cKDTree 

def fit_sphere_least_squares(pts):
    if len(pts) < 15: return None, None
    A = np.zeros((len(pts), 4))
    A[:, 0:3] = pts
    A[:, 3] = 1
    B = np.sum(pts**2, axis=1)
    try:
        res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        center = res[0:3] / 2
        r2 = res[3] + np.sum(center**2)
        if r2 < 0: return None, None
        return center, np.sqrt(r2)
    except:
        return None, None



def get_curvature_mask(pts, search_radius, curvature_thresh=0.06):
    """
    通过局部协方差矩阵计算特征值，提取表面平滑区域的 Mask，
    """
    tree = cKDTree(pts)
    neighbors_list = tree.query_ball_point(pts, r=search_radius, workers=-1)
    
    valid_mask = np.ones(len(pts), dtype=bool)
    
    for i, neighbors in enumerate(neighbors_list):
        if len(neighbors) < 5:
            valid_mask[i] = False
            continue
            
        local_pts = pts[neighbors]
        
        centroid = np.mean(local_pts, axis=0)
        centered_pts = local_pts - centroid
        
        cov_matrix = np.dot(centered_pts.T, centered_pts) / len(neighbors)
        
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        
        sum_eigen = np.sum(eigenvalues)
        if sum_eigen <= 1e-6:
            continue
            
        # pca曲率
        curvature = eigenvalues[0] / sum_eigen
        
        # 如果pca曲率大于阈值，说明处于缝隙
        if curvature > curvature_thresh:
            valid_mask[i] = False
            
    return valid_mask


def segment_and_filter_apples(pts_3d, cur_thresh, min_radius, max_radius):
    red_pcd = o3d.geometry.PointCloud()
    red_pcd.points = o3d.utility.Vector3dVector(pts_3d)

    # 下采样减少计算压力
    red_pcd = red_pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    if len(red_pcd.points) < 20: return []
    
    pts = np.asarray(red_pcd.points)
    
    # pca曲率分析
    valid_mask = get_curvature_mask(pts, search_radius=KDTREE_SEARCH_RADIUS, curvature_thresh=cur_thresh)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 20: return []
    filtered_pcd = red_pcd.select_by_index(valid_indices)


    # DBSCAN 聚类
    labels = np.array(filtered_pcd.cluster_dbscan(eps=DBSCAN_EPS, min_points=DBSCAN_MIN_POINTS))
    if len(labels) == 0 or labels.max() < 0: return []
    
    results = []
    for c in range(labels.max() + 1):
        cluster_idx = np.where(labels == c)[0]
        if len(cluster_idx) < SPHERE_FIT_MIN_PTS: 
            continue
        cluster_pts = np.asarray(filtered_pcd.points)[cluster_idx]
        center, radius = fit_sphere_least_squares(cluster_pts)
        
        if center is not None and min_radius < radius < max_radius:
            results.append((center, radius, cluster_pts))
    
    
    # 合并相近的球体
    merged_results = []
    
    for curr_center, curr_radius, curr_pts in results:
        found_match = False
        
        for i, (merged_center, merged_radius, merged_pts) in enumerate(merged_results):

            center_dist = np.linalg.norm(curr_center - merged_center)
            
            if center_dist < max(curr_radius, merged_radius) * 0.8:
                
                combined_pts = np.vstack((merged_pts, curr_pts))
                
                new_center, new_radius = fit_sphere_least_squares(combined_pts)
                
                if new_center is not None:
                    if new_radius < max_radius:
                        merged_results[i] = (new_center, new_radius, combined_pts)
                    else:
                        # 如果合并后算出超大半径，则只保留点云数量多的那一块碎片
                        if len(curr_pts) > len(merged_pts):
                            merged_results[i] = (curr_center, curr_radius, curr_pts)
                
                found_match = True
                break 
                
        if not found_match:
            merged_results.append((curr_center, curr_radius, curr_pts))
            
    return merged_results