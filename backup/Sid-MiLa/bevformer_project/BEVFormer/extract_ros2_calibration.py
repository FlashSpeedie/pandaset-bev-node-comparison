import sqlite3
import json

# Replace this with the path to one of Dr. Xu's .db3 bag files if you have them
bag_file = 'path/to/your/rosbag.db3' 

try:
    conn = sqlite3.connect(bag_file)
    cursor = conn.cursor()
    
    # Look for camera_info topics
    cursor.execute("SELECT name FROM topics WHERE name LIKE '%camera_info%'")
    topics = cursor.fetchall()
    
    if not topics:
        print("No camera_info topics found in this bag.")
    else:
        print(f"Found camera info topics: {topics}")
        print("The intrinsics (K matrix) are inside these topics!")
        
except Exception as e:
    print(f"Could not read bag: {e}")
