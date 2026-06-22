import open3d as o3d
import numpy as np

# 1. 加载点云（PLY/PCD等格式）
pcd = o3d.io.read_point_cloud("zhui.ply")

# 2. 执行泊松重建（最快速且效果好的方法）
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=9  # 控制重建精度，值越大越精细但越慢
)

# 3. 保存为OBJ格式
o3d.io.write_triangle_mesh("output.obj", mesh)

print("点云已成功重建为OBJ模型！")
