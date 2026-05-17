import json
import random
import matplotlib.pyplot as plt

# Function to calculate Manhattan distance
def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Function to calculate latencies to all sites
def get_all_latencies(sites, user_coord, user):
    return {f"('{site}', '{user}')": manhattan_distance(user_coord, coord) for site, coord in sites.items()}

# Function to find the nearest site distance
def find_nearest_distance(sites, user_coord):
    return min(manhattan_distance(user_coord, coord) for coord in sites.values())

# Function to generate a valid user coordinate
def generate_user_coordinate(cloud_sites, edge_sites, existing_users):
    while True:
        x, y = random.randint(0, 99), random.randint(0, 99)
        dist_to_cloud = find_nearest_distance(cloud_sites, (x, y))
        dist_to_edge = find_nearest_distance(edge_sites, (x, y))
        
        if 30 <= dist_to_cloud <= 60 and 5 <= dist_to_edge <= 20:
            # Check distance to all existing users
            if all(manhattan_distance((x, y), coord) > 25 for coord in existing_users.values()):
                return (x, y)

# Function to visualize the sites and users
def visualize_results(cloud_sites, edge_sites, users):
    plt.figure(figsize=(10, 10))
    
    # Plot cloud sites (Red Circles)
    for name, (x, y) in cloud_sites.items():
        plt.scatter(x, y, color="red", marker="o", s=100, label="Cloud Site" if name == "c1" else "")
        plt.text(x, y, name, fontsize=12, ha='right', va='bottom')

    # Plot edge sites (Blue Squares)
    for name, (x, y) in edge_sites.items():
        plt.scatter(x, y, color="blue", marker="s", s=100, label="Edge Site" if name == "e1" else "")
        plt.text(x, y, name, fontsize=12, ha='right', va='bottom')

    # Plot users (Green Triangles)
    for name, (x, y) in users.items():
        plt.scatter(x, y, color="green", marker="^", s=80, label="User" if name == "u1" else "")
        plt.text(x, y, name, fontsize=10, ha='left', va='top')

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Cloud, Edge, and User Locations")
    plt.legend()
    
    # Save the plot
    plt.savefig("network_visualization.png")
    plt.show()

# Main function
def main():
    # Define cloud and edge site coordinates
    cloud_sites = {"c1": (50, 50)}
    edge_sites = {"e1": (25, 75), "e2": (25, 25), "e3": (75, 75), "e4": (75, 25)}
    num_users = 10  # Number of users to generate

    # Merge cloud and edge sites into one dictionary
    all_sites = {**cloud_sites, **edge_sites}

    # Generate user data
    users = {}
    latency_matrix = {}

    for i in range(1, num_users + 1):
        user_name = f"u{i}"
        user_coord = generate_user_coordinate(cloud_sites, edge_sites, users)
        users[user_name] = user_coord
        latency_matrix.update(get_all_latencies(all_sites, user_coord, user_name))

    # Create final JSON structure
    result = {
        "users": users,
        "latency_site_user": latency_matrix
    }

    # Save to JSON file with tighter formatting for user coordinates
    with open("network_latencies.json", "w") as json_file:
        json.dump(result, json_file, indent=4, separators=(',', ': '))

    print("JSON file 'network_latencies.json' created successfully!")
    
    # Generate visualization
    visualize_results(cloud_sites, edge_sites, users)

if __name__ == "__main__":
    main()