import open3d as o3d
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.interpolate import griddata
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import pyshtools as pysh
import os
import pandas as pd


# Compute volume using Convex Hull
def compute_volume(pcd):
    # Convert the point cloud to a NumPy array
    points = np.asarray(pcd.points)
    if len(points) < 4:  # ConvexHull requires at least 4 non-coplanar points
        print("Not enough points for Convex Hull.")
        return None
    try:
        hull = ConvexHull(points)
        volume = hull.volume
    except Exception as e:
        print("Error computing Convex Hull:", e)
        volume = None
    return volume

# Compute AABB-based diameter
def compute_aabb(pcd):
    aabb = pcd.get_axis_aligned_bounding_box()
    min_bound = aabb.get_min_bound()
    max_bound = aabb.get_max_bound()
    dimensions = max_bound - min_bound
    length, width, height = dimensions
    return length, width, height

# Compute PCA-based diameter
def compute_pca_diameter(pcd):
    points = np.asarray(pcd.points)
    pca = PCA(n_components=1)
    pca.fit(points)
    projections = points @ pca.components_[0]
    min_idx, max_idx = np.argmin(projections), np.argmax(projections)
    min_point, max_point = points[min_idx], points[max_idx]
    return np.linalg.norm(max_point - min_point), min_point, max_point

def compute_stalk_point(pcd, radius=0.08):
    """
    Compute the stalk point based on the center of gravity and a defined radius.
    
    Parameters:
    - pcd (open3d.geometry.PointCloud): The point cloud of the apple.
    - radius (float): Radius around the center of gravity to search for the stalk point.

    Returns:
    - stalk_point (numpy.ndarray or None): The coordinates of the stalk point.
    - center_of_gravity (numpy.ndarray): The center of gravity of the apple.
    """
    # Convert the point cloud to a NumPy array
    points = np.asarray(pcd.points)

    pca_diameter, _, _ = compute_pca_diameter(pcd)
    radius = radius * pca_diameter
    print("Dynamic Radius Stalk:", radius*pca_diameter)
    
    # Calculate the center of gravity
    center_of_gravity = np.mean(points, axis=0)

    # Filter points near the center in the XY plane, within the specified radius
    xy_distances = np.linalg.norm(points[:, :2] - center_of_gravity[:2], axis=1)
    nearby_points = points[xy_distances < radius]

    # Identify points in the upper half (above the center of gravity)
    upper_half_points = nearby_points[nearby_points[:, 2] > center_of_gravity[2]]

    if len(upper_half_points) > 0:
        # Find the point with the lowest Z-coordinate (stalk point)
        stalk_point = upper_half_points[np.argmin(upper_half_points[:, 2])]
    else:
        stalk_point = None
        print("No points found in the upper half for stalk calculation.")

    return stalk_point

def compute_calyx_point(pcd, cog, radius_factor=0.08):
    """
    Compute the calyx point and detect if it has a crown or an indentation.

    Parameters:
    - pcd (open3d.geometry.PointCloud): The point cloud of the apple.
    - cog (numpy.ndarray): Center of gravity of the apple.
    - radius_factor (float): Fraction of PCA diameter to use as the radius for the calyx region.
    - base_radius (float): Minimum radius for the calyx region.

    Returns:
    - calyx_point (numpy.ndarray or None): The detected calyx point.
    - has_crown (bool): True if the calyx has a crown, False if it has an indentation.
    - dynamic_radius (float): The radius used for the calyx region.
    - calyx_region (numpy.ndarray): Points inside the calyx region.
    - initial_calyx_point (numpy.ndarray): The initial calyx point (lowest or highest Z).
    """
    points = np.asarray(pcd.points)

    # Compute the PCA diameter for dynamic radius
    pca_diameter, _, _ = compute_pca_diameter(pcd)
    dynamic_radius = radius_factor * pca_diameter
    print("Dynamic Radius Calyx:", dynamic_radius)

    # Define the calyx region: below the COG and within the radius
    for attempt in range(3):  # Try up to 3 radius adjustments
        calyx_region_mask = (
            (np.linalg.norm(points[:, :2] - cog[:2], axis=1) <= dynamic_radius) &  # Within radius
            (points[:, 2] <= cog[2])  # Below the COG
        )
        calyx_region = points[calyx_region_mask]

        if len(calyx_region) > 0:
            break  # Exit loop if points are found
        dynamic_radius *= 1.5  # Increase radius dynamically

    if len(calyx_region) == 0:
        print("No points found in the calyx region.")
        return None, False, dynamic_radius, np.array([])
    
    ########### NEW ###########
    # Sort the calyx region by Z-coordinate
    sorted_calyx_region = calyx_region[np.argsort(calyx_region[:, 2])]
    print("Length of Sorted Calyx Region:", len(sorted_calyx_region))
    print(calyx_region.shape)

    # Discard the lowest 25% of points by Z
    discard_count = int(len(sorted_calyx_region) * 0.025)
    filtered_calyx_region = sorted_calyx_region[discard_count:]
    print("Length of Filtered Calyx Region:", len(filtered_calyx_region))
    print(filtered_calyx_region.shape)

    if len(filtered_calyx_region) == 0:
        print("No points found in the filtered calyx region.")
        return None, False, dynamic_radius, np.array([])
    #########################

    # Find the point with the lowest Z-coordinate in the calyx region
    initial_calyx_point = filtered_calyx_region[np.argmin(filtered_calyx_region[:, 2])]

    # Define the surrounding ring
    ring_radius = dynamic_radius * 1.5  # Slightly larger radius
    ring_mask = (
        (np.linalg.norm(points[:, :2] - cog[:2], axis=1) > dynamic_radius) &  # Outside calyx region
        (np.linalg.norm(points[:, :2] - cog[:2], axis=1) <= ring_radius)  # Within ring radius
    )
    ring_points = points[ring_mask]

    # Check if any ring points have a lower Z-coordinate than the initial calyx point
    if len(ring_points) > 0 and np.min(ring_points[:, 2]) < initial_calyx_point[2]:
        has_crown = False  # Indentation detected
        calyx_point = calyx_region[np.argmax(calyx_region[:, 2])]  # Highest point in calyx region
    else:
        has_crown = True  # Crown detected
        calyx_point = initial_calyx_point  # Lowest point in calyx region

    return calyx_point, has_crown, dynamic_radius, filtered_calyx_region, initial_calyx_point



def find_rim_points_stalk(pcd, stalk_point, cog, radius_factor=0.2, num_segments=24, exclude_top_n=5):
    """
    Identify the rim points around the stalk region of the apple point cloud.

    Parameters:
    - pcd (open3d.geometry.PointCloud): The point cloud of the apple.
    - stalk_point (numpy.ndarray): The 3D coordinates of the stalk point.
    - center_of_gravity (numpy.ndarray): The 3D coordinates of the apple's center of gravity.
    - radius (float): The radius around the pit point to search for rim points.
    - num_segments (int): The number of angular segments to divide the neighborhood into.
    - exclude_top_n (int): The number of highest Z-points to exclude (to avoid outliers from the stalk).

    Returns:
    - rim_points (numpy.ndarray): An array of 3D coordinates representing the rim points.
    """
    # Convert the point cloud to a NumPy array
    points = np.asarray(pcd.points)

    pca_diameter, _, _ = compute_pca_diameter(pcd)
    radius = pca_diameter * radius_factor
    
    # Compute the 2D distances (in the XY plane) from each point to the pit point
    xy_distances = np.linalg.norm(points[:, :2] - stalk_point[:2], axis=1)

    # Select points within the specified radius around the pit point
    neighborhood_points = points[xy_distances < radius]

    # Calculate the angle of each neighborhood point relative to the pit point in the XY plane
    angles = np.arctan2(neighborhood_points[:, 1] - stalk_point[1], neighborhood_points[:, 0] - stalk_point[0])
    
    rim_points = []  # Initialize an empty list to store the rim points

    # Divide the neighborhood into angular segments and find a representative point in each segment
    for i in range(num_segments):
        # Define the angular range for the current segment
        angle_min = -np.pi + (i * 2 * np.pi / num_segments)
        angle_max = -np.pi + ((i + 1) * 2 * np.pi / num_segments)

        # Filter points that fall within the current angular segment
        segment_points = neighborhood_points[(angles >= angle_min) & (angles < angle_max)]

        if len(segment_points) == 0:  # Skip empty segments
            continue

        # Exclude the highest Z-points (outliers) in the stalk area
        if stalk_point[2] > cog[2]:
            segment_points = segment_points[np.argsort(segment_points[:, 2])[:-exclude_top_n]]

        # Select the highest point for the stalk area
        rim_point = segment_points[np.argmax(segment_points[:, 2])]
        rim_points.append(rim_point)

    rim_points = np.array(rim_points)
    distances = np.linalg.norm(rim_points - stalk_point, axis=1)
    sorted_indices = np.argsort(distances)
    sorted_rim_points = rim_points[sorted_indices]
    rim_points = sorted_rim_points[4:-4]

    return rim_points

def find_rim_points_calyx(pcd, calyx_point, has_crown, cog, radius_factor=0.2, num_segments=24):
    """
    Find rim points for the calyx based on whether it has a crown or an indentation.

    Parameters:
    - pcd (open3d.geometry.PointCloud): The full point cloud of the apple.
    - calyx_point (numpy.ndarray): The detected calyx point (lowest or highest Z).
    - has_crown (bool): True if the calyx has a crown, False otherwise.
    - cog (numpy.ndarray): The center of gravity of the apple.
    - radius_factor (float): The radius factor relative to the PCA diameter for the calyx region.
    - num_segments (int): Number of angular segments to divide the region into.

    Returns:
    - rim_points (numpy.ndarray): The detected rim points around the calyx.
    """
    points = np.asarray(pcd.points)
    pca_diameter, _, _ = compute_pca_diameter(pcd)
    outer_radius = pca_diameter * radius_factor
    inner_radius = outer_radius / 2  # Smaller radius for crown or inner exclusion for indentation

    if calyx_point is None:
        print("Calyx point is None. Cannot compute rim points.")
        return np.array([])

    # Compute distances in the XY plane from the calyx point
    xy_distances = np.linalg.norm(points[:, :2] - calyx_point[:2], axis=1)

    # Define the neighborhood based on crown/indentation logic
    if has_crown:
        # Narrow search for crown region (within half the radius)
        neighborhood_mask = (
            (xy_distances <= inner_radius) &        # Within narrow crown radius
            (points[:, 2] <= cog[2]) &             # Below the COG
            (points[:, 2] > calyx_point[2])        # Higher than calyx_point for crown
        )
    else:
        # Exclude inner region and search within outer radius for indentation
        neighborhood_mask = (
            (xy_distances > inner_radius) &        # Outside inner radius
            (xy_distances <= outer_radius) &       # Within outer radius
            (points[:, 2] <= cog[2]) &             # Below the COG
            (points[:, 2] < calyx_point[2])        # Lower than calyx_point for indentation
        )

    neighborhood_points = points[neighborhood_mask]

    if len(neighborhood_points) == 0:
        print("No points found in the neighborhood for calyx rim.")
        return np.array([])

    # Calculate angles relative to the calyx point
    angles = np.arctan2(neighborhood_points[:, 1] - calyx_point[1], neighborhood_points[:, 0] - calyx_point[0])

    rim_points = []
    for i in range(num_segments):
        # Define the angular range for this segment
        min_angle = -np.pi + i * 2 * np.pi / num_segments
        max_angle = -np.pi + (i + 1) * 2 * np.pi / num_segments

        # Filter points in the current segment
        segment_points = neighborhood_points[(angles >= min_angle) & (angles < max_angle)]

        if len(segment_points) == 0:
            continue  # Skip empty segments

        # Find rim points based on crown/indentation logic
        if has_crown:
            rim_point = segment_points[np.argmax(segment_points[:, 2])]  # Highest Z for crown
        else:
            rim_point = segment_points[np.argmin(segment_points[:, 2])]  # Lowest Z for indentation
        rim_points.append(rim_point)

    rim_points = np.array(rim_points)
    distances = np.linalg.norm(rim_points[:, :2] - calyx_point[:2], axis=1)
    sorted_indices = np.argsort(distances)
    sorted_rim_points = rim_points[sorted_indices]
    rim_points = sorted_rim_points[4:-4]

    return rim_points


# Fit plane using RANSAC
def fit_plane_ransac(points):
    X = points[:, :2]
    Z = points[:, 2]
    ransac = RANSACRegressor()
    ransac.fit(X, Z)
    a, b = ransac.estimator_.coef_
    normal = np.array([-a, -b, 1])
    normal /= np.linalg.norm(normal)
    plane_point = points.mean(axis=0)
    return plane_point, normal

# Project point onto a plane
def project_point_onto_plane(point, plane_point, plane_normal):
    vector_to_plane = point - plane_point
    distance = np.dot(vector_to_plane, plane_normal)
    projected_point = point - distance * plane_normal
    return projected_point, abs(distance)

# Compute pit width statistics
def compute_pit_width_stats(rim_points, pit_point):
    """
    Computes the mean diameter and variability (standard deviation) of distances from rim points to the central pit point.

    Parameters:
    - rim_points (numpy.ndarray): An array of 3D coordinates representing the rim points.
    - pit_point (numpy.ndarray): The 3D coordinates of the central stalk or calyx point.

    Returns:
    - mean_diameter (float): The mean diameter of the pit (2 * mean radius).
    - std_distance (float): The standard deviation of distances from rim points to the pit point.
    """
    if len(rim_points) == 0:
        return None, None

    # Compute distances in the XY plane (radius)
    xy_distances = np.linalg.norm(rim_points[:, :2] - pit_point[:2], axis=1)
    
    # Calculate the mean radius and standard deviation
    mean_radius = np.mean(xy_distances)
    std_distance = np.std(xy_distances)
    
    # Convert radius to diameter
    mean_diameter = 2 * mean_radius
    
    return mean_diameter, std_distance

# Compute projected height
def compute_projected_height(stalk_projected, calyx_projected):
    return np.linalg.norm(stalk_projected - calyx_projected)

# Compute SPHARM features
def compute_spharm_features_from_pcd(pcd, grid_resolution=120, l_max=10):
    points = np.asarray(pcd.points)
    
    # Translation normalization (center of gravity)
    cog = points.mean(axis=0)
    shifted_points = points - cog
    
    # Scaling normalization (scale to unit sphere)
    max_radius = np.sqrt((shifted_points ** 2).sum(axis=1)).max()  # Maximum distance from origin
    normalized_points = shifted_points / max_radius
    
    # Convert to spherical coordinates
    x, y, z = normalized_points[:, 0], normalized_points[:, 1], normalized_points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    
    # Interpolate radial distances onto a spherical grid
    nlat, nlon = grid_resolution, 2 * grid_resolution
    theta_grid = np.linspace(0, np.pi, nlat)
    phi_grid = np.linspace(0, 2 * np.pi, nlon)
    theta_mesh, phi_mesh = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    r_grid = griddata((theta, phi), r, (theta_mesh.flatten(), phi_mesh.flatten()), method='linear', fill_value=0).reshape(nlat, nlon)
    
    # Expand into spherical harmonics
    grid = pysh.SHGrid.from_array(r_grid)
    clm = pysh.expand.SHExpandDH(grid.data, sampling=2, lmax_calc=l_max)
    coeffs = pysh.SHCoeffs.from_array(clm)
    
    return coeffs.spectrum()

# Compute color features
def compute_color_features(pcd):
    colors = np.asarray(pcd.colors)
    return colors.mean(axis=0), colors.std(axis=0), colors.ptp(axis=0)

def compute_xy_distances_to_calyx(rim_points, calyx_point):
    """
    Compute statistics of XY distances between rim points and the calyx point.

    Parameters:
    - rim_points (numpy.ndarray): The 3D coordinates of the rim points.
    - calyx_point (numpy.ndarray): The central calyx point for reference.

    Returns:
    - xy_distance_stats (dict): Dictionary containing mean, std, min, and max distances.
    """
    if len(rim_points) == 0:
        return {"xy_mean_dist": 0, "xy_std_dist": 0, "xy_min_dist": 0, "xy_max_dist": 0}

    # Compute distances in the XY plane
    xy_distances = np.linalg.norm(rim_points[:, :2] - calyx_point[:2], axis=1)

    # Extract statistics
    xy_distance_stats = {
        "xy_mean_dist": np.mean(xy_distances),
        "xy_std_dist": np.std(xy_distances),
        "xy_min_dist": np.min(xy_distances),
        "xy_max_dist": np.max(xy_distances),
    }
    return xy_distance_stats


def compute_z_height_differences(rim_points, calyx_point):
    """
    Compute statistics of Z-height differences between consecutive rim points 
    and classify the flatness of the calyx region.

    Parameters:
    - rim_points (numpy.ndarray): The 3D coordinates of the rim points.
    - calyx_point (numpy.ndarray): The central calyx point for reference.

    Returns:
    - z_height_stats (dict): Dictionary containing mean, std, max Z-height differences, 
      and flatness category.
    """
    if len(rim_points) < 2:
        return {
            "z_mean_diff": 0,
            "z_std_diff": 0,
            "flatness_category": None  # Default if no rim points
        }

    # Sort rim points by their angle around the calyx point
    angles = np.arctan2(rim_points[:, 1] - calyx_point[1], rim_points[:, 0] - calyx_point[0])
    sorted_indices = np.argsort(angles)
    sorted_rim_points = rim_points[sorted_indices]

    # Compute Z-height differences
    z_differences = np.diff(sorted_rim_points[:, 2])

    # Compute flatness based on normalized mean Z-height differences
    z_absolute_differences = np.abs(sorted_rim_points[:, 2] - calyx_point[2])
    mean_z_diff = np.mean(z_absolute_differences)
    max_z_diff = np.max(z_absolute_differences)

    # Normalize mean_z_diff by max_z_diff to account for scale differences
    normalized_mean_z_diff = mean_z_diff / max_z_diff if max_z_diff != 0 else 0

    # Define flatness categories based on normalized mean Z-difference
    if normalized_mean_z_diff < 0.2:
        flatness_category = 1  # Very flat
    elif normalized_mean_z_diff < 0.4:
        flatness_category = 2  # Flat
    elif normalized_mean_z_diff < 0.6:
        flatness_category = 3  # Moderately irregular
    elif normalized_mean_z_diff < 0.8:
        flatness_category = 4  # Irregular
    else:
        flatness_category = 5  # Very irregular

    # Extract statistics
    z_height_stats = {
        "z_mean_diff": np.mean(np.abs(z_differences)),  # Average absolute difference
        "z_std_diff": np.std(z_differences),           # Standard deviation of differences
        "flatness_category": flatness_category         # Flatness category
    }
    return z_height_stats

# Process PCDs and extract features
def process_pcds_and_extract_features(base_folder, variety_mapping_file, output_file):
    mapping_df = pd.read_csv(variety_mapping_file)
    mapping_dict = mapping_df.set_index('SACCBaum')['PL.Code'].to_dict()
    all_data = []

    for tree_folder in os.listdir(base_folder):
        tree_path = os.path.join(base_folder, tree_folder)
        if not os.path.isdir(tree_path):
            continue

        for apple_folder in os.listdir(tree_path):
            apple_path = os.path.join(tree_path, apple_folder)
            if not os.path.isdir(apple_path):
                continue

            pcd_file = os.path.join(apple_path, 'final_cleaned_pcd.pcd')
            if not os.path.exists(pcd_file):
                print(f"PCD file not found for {apple_folder}")
                continue

            pcd = o3d.io.read_point_cloud(pcd_file)
            if pcd.is_empty():
                print(f"Empty PCD file for {apple_folder}")
                continue

            try:
                points = np.asarray(pcd.points)
                cog = points.mean(axis=0)
                volume = compute_volume(pcd)
                aabb_length, aabb_width, aabb_height = compute_aabb(pcd)
                diameter_pca, _, _ = compute_pca_diameter(pcd)

                stalk_point = compute_stalk_point(pcd)
                stalk_rim_points = find_rim_points_stalk(pcd, stalk_point, cog)

                calyx_point, has_crown, dynamic_radius, calyx_region, intial_calyx_point = compute_calyx_point(pcd, cog)
                calyx_rim_points = find_rim_points_calyx(pcd, calyx_point, has_crown, cog)

                distance = np.linalg.norm(stalk_point - calyx_point)

                # Z-Height Differences for Calyx Rim Points
                z_height_features = compute_z_height_differences(calyx_rim_points, calyx_point)

                # XY Distances for Calyx Rim Points
                xy_distance_features = compute_xy_distances_to_calyx(calyx_rim_points, calyx_point)

                stalk_plane, stalk_normal = fit_plane_ransac(stalk_rim_points)
                calyx_plane, calyx_normal = fit_plane_ransac(calyx_rim_points)

                stalk_projected, stalk_depth = project_point_onto_plane(stalk_point, stalk_plane, stalk_normal)
                calyx_projected, calyx_depth = project_point_onto_plane(calyx_point, calyx_plane, calyx_normal)
                projected_height = compute_projected_height(stalk_projected, calyx_projected)

                stalk_width, stalk_std = compute_pit_width_stats(stalk_rim_points, stalk_point)
                calyx_width, calyx_std = compute_pit_width_stats(calyx_rim_points, calyx_point)

                stalk_pit_ratio = stalk_depth / stalk_width if stalk_width > 0 else None
                calyx_pit_ratio = calyx_depth / calyx_width if calyx_width > 0 else None

                spharm_features = compute_spharm_features_from_pcd(pcd)
                mean_color, std_color, color_range = compute_color_features(pcd)

                variety = mapping_dict.get(tree_folder, None)

                #ratios
                calyx_to_apple = calyx_width / diameter_pca if diameter_pca > 0 else None
                stalk_to_apple = stalk_width / diameter_pca if diameter_pca > 0 else None
                height_to_width = projected_height / diameter_pca if diameter_pca > 0 else None

                apple_data = {
                    'Variety': variety, 'Tree': tree_folder, 'Apple': apple_folder, 'Volume': volume,
                    'AABB_Length': aabb_length, ' AABB_Width': aabb_width, 'AABB_Height': aabb_height, 'PCA_Diameter': diameter_pca,
                    'Stalk_Depth': stalk_depth, 'Calyx_Depth': calyx_depth,
                    'Stalk_Width': stalk_width, 'Calyx_Width': calyx_width,
                    'Projected_Height': projected_height, 'Stalk_To_Calyx_Distance': distance,
                    'Stalk_Pit_Ratio': stalk_pit_ratio, 'Calyx_Pit_Ratio': calyx_pit_ratio,
                    'Mean_Color': mean_color, 'Std_Color': std_color, 'Color_Range': color_range,
                    'SPHARM_Features': spharm_features,
                    'Z_Mean_Diff': z_height_features["z_mean_diff"],
                    'Z_Std_Diff': z_height_features["z_std_diff"],
                    'XY_Mean_Dist': xy_distance_features["xy_mean_dist"],
                    'XY_Std_Dist': xy_distance_features["xy_std_dist"],
                    'Calyx_to_Apple_Ratio': calyx_to_apple, 'Stalk_to_Apple_Ratio': stalk_to_apple, 'Height_to_Width_Ratio': height_to_width
                }
                all_data.append(apple_data)

            except Exception as e:
                print(f"Error processing {apple_folder}: {e}")
                continue

    df = pd.DataFrame(all_data)


    spharm_cols = [f"SPHARM_Feature_{i}" for i in range(len(df['SPHARM_Features'][0]))]
    for i, col in enumerate(spharm_cols):
        df[col] = df['SPHARM_Features'].apply(lambda x: x[i])
    df.drop(columns=['SPHARM_Features'], inplace=True)

    # Expand Color Features
    color_features = ['Mean_Color', 'Std_Color', 'Color_Range']
    for feature in color_features:
        if isinstance(df[feature].iloc[0], (np.ndarray, list)):
            num_channels = len(df[feature].iloc[0])
            for i in range(num_channels):
                df[f"{feature}_{i}"] = df[feature].apply(lambda x: x[i])
            df.drop(columns=[feature], inplace=True)

    scaler = MinMaxScaler()
    feature_cols = [col for col in df.columns if col not in ['Tree', 'Apple', 'Variety', 'Flatness_Category']]
    for col in feature_cols:
        if isinstance(df[col].iloc[0], (list, np.ndarray)):
            print(f"Non-scalar column detected: {col}")
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

# Generate boxplots
def generate_boxplots(feature_file, output_folder):
    df = pd.read_csv(feature_file)
    features = [col for col in df.columns if col not in ['Tree', 'Apple', 'Variety', 'Has_Crown', 'Flatness_Category']]
    os.makedirs(output_folder, exist_ok=True)

    for feature in features:
        plt.figure(figsize=(10, 6))
        df.boxplot(column=feature, by='Variety', grid=False)
        plt.title(f'{feature} by Variety')
        plt.suptitle('')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(feature)
        plt.savefig(os.path.join(output_folder, f'{feature}_boxplot.png'))
        plt.close()