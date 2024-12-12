import os
import open3d as o3d
import numpy as np
from utils_FE import (compute_aabb, compute_volume, compute_color_features, compute_projected_height, compute_pca_diameter,
                        compute_calyx_point, compute_stalk_point, find_rim_points_calyx, find_rim_points_stalk, fit_plane_ransac, project_point_onto_plane,
                        compute_projected_height, compute_pit_width_stats, compute_spharm_features_from_pcd)

# Define the base directory containing the PCD files
base_folder_path = "C:/Daten/PA2/Code/pcds_proc"


# Iterate through each tree folder (varieties)
for tree_folder in os.listdir(base_folder_path):
    print(f"Processing tree folder: {tree_folder}")
    tree_path = os.path.join(base_folder_path, tree_folder)
    if not os.path.isdir(tree_path):
        continue

    # Iterate through each apple folder in the tree folder
    for apple_folder in os.listdir(tree_path):
        apple_path = os.path.join(tree_path, apple_folder)
        if not os.path.isdir(apple_path):
            continue

        # Load the cleaned point cloud file
        pcd_file = os.path.join(apple_path, 'spherical80_cleaned_pcd.pcd')
        if not os.path.exists(pcd_file):
            print(f"PCD file not found for {apple_folder}")
            continue

        # Load the point cloud
        pcd = o3d.io.read_point_cloud(pcd_file)

        if pcd.is_empty():
            print(f"Empty point cloud for {apple_folder}")
            continue

        print(f"Visualizing tree{tree_folder},apple: {apple_folder}...")

        points = np.asarray(pcd.points)
        cog = points.mean(axis=0)
        volume = compute_volume(pcd)
        aabb_length, aabb_width, aabb_height = compute_aabb(pcd)
        diameter_pca, _, _ = compute_pca_diameter(pcd)
        print(f"Diameter: {diameter_pca}, Type: {type(diameter_pca)}")
        stalk_point = compute_stalk_point(pcd)
        # print(f"Stalk point: {stalk_point}")
        stalk_rim_points = find_rim_points_stalk(pcd, stalk_point, cog)
        calyx_point, has_crown, calyx_radius, calyx_cluster, initial_calyx_point = compute_calyx_point(pcd, cog)
        print(has_crown)
        calyx_rim_points = find_rim_points_calyx(pcd, calyx_point, has_crown, cog)
        print(f"Calyx rim points found: {len(calyx_rim_points)}")

        distance = np.linalg.norm(stalk_point - calyx_point)

        stalk_plane, stalk_normal = fit_plane_ransac(stalk_rim_points)
        calyx_plane, calyx_normal = fit_plane_ransac(calyx_rim_points)
        stalk_projected, stalk_depth = project_point_onto_plane(stalk_point, stalk_plane, stalk_normal)
        calyx_projected, calyx_depth = project_point_onto_plane(calyx_point, calyx_plane, calyx_normal)
        projected_height = compute_projected_height(stalk_projected, calyx_projected)
        print("Stalk depth: ", stalk_depth,"Calyx depth: ", calyx_depth)

        stalk_width, stalk_std = compute_pit_width_stats(stalk_rim_points, stalk_point)
        calyx_width, calyx_std = compute_pit_width_stats(calyx_rim_points, calyx_point)

        stalk_pit_ratio = stalk_depth / stalk_width if stalk_width > 0 else None
        calyx_pit_ratio = calyx_depth / calyx_width if calyx_width > 0 else None

        # spharm_features = compute_spharm_features_from_pcd(pcd)
        mean_color, std_color, color_range = compute_color_features(pcd)

        calyx_spheres = []
        for point in calyx_cluster:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0003)
            sphere.translate(point)
            sphere.paint_uniform_color([1, 0, 0])
            calyx_spheres.append(sphere)

        #################################

        # Create spheres for the rim points
        rim_spheres = []
        # for point in stalk_rim_points:
        #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        #     sphere.translate(point)
        #     sphere.paint_uniform_color([0, 1, 1])  # Cyan for stalk rim points
        #     rim_spheres.append(sphere)

        for point in calyx_rim_points:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            sphere.translate(point)
            sphere.paint_uniform_color([0, 1, 1])  # Purple for calyx rim points
            rim_spheres.append(sphere)

        # Create spheres for the COG, stalk, and calyx points
        cog_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.002)
        cog_sphere.translate(cog)
        cog_sphere.paint_uniform_color([1, 0, 0])

        stalk_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        stalk_sphere.translate(stalk_point)
        stalk_sphere.paint_uniform_color([0.5, 0, 1])

        calyx_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        calyx_sphere.translate(calyx_point)
        calyx_sphere.paint_uniform_color([0.5, 0, 1])

        # Create lines for pit widths
        stalk_width_line = o3d.geometry.LineSet()
        stalk_width_line.points = o3d.utility.Vector3dVector([stalk_point, stalk_projected])
        stalk_width_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        stalk_width_line.colors = o3d.utility.Vector3dVector([[0, 1, 1]])

        calyx_width_line = o3d.geometry.LineSet()
        calyx_width_line.points = o3d.utility.Vector3dVector([calyx_point, calyx_projected])
        calyx_width_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        calyx_width_line.colors = o3d.utility.Vector3dVector([[0.5, 0, 1]])

        # Create lines for pit heights
        stalk_height_line = o3d.geometry.LineSet()
        stalk_height_line.points = o3d.utility.Vector3dVector([stalk_point, stalk_projected])
        stalk_height_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        stalk_height_line.colors = o3d.utility.Vector3dVector([[0.5, 0, 1]])

        calyx_height_line = o3d.geometry.LineSet()
        calyx_height_line.points = o3d.utility.Vector3dVector([calyx_point, calyx_projected])
        calyx_height_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        calyx_height_line.colors = o3d.utility.Vector3dVector([[0.5, 0, 1]])

        diameter, p1, p2 = compute_pca_diameter(pcd)

        # Create line for diameter
        diameter_line = o3d.geometry.LineSet()
        diameter_line.points = o3d.utility.Vector3dVector([p1, p2])
        diameter_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        diameter_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])


        # Create line for projected height
        line_set_height = o3d.geometry.LineSet()
        line_set_height.points = o3d.utility.Vector3dVector([stalk_projected, calyx_projected])
        line_set_height.lines = o3d.utility.Vector2iVector([[0, 1]])
        line_set_height.colors = o3d.utility.Vector3dVector([[1, 0, 0]])

        # Create black sphere for inital calyx point
        initial_calyx_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        initial_calyx_sphere.translate(initial_calyx_point)
        initial_calyx_sphere.paint_uniform_color([0, 0, 0])



        # Coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

        o3d.visualization.draw_geometries(
            [pcd, calyx_sphere,  calyx_height_line] + rim_spheres,
            window_name=f"Apple Visualization: {apple_folder}",
            width=1600,
            height=1200
        )

print("All visualizations completed.")
