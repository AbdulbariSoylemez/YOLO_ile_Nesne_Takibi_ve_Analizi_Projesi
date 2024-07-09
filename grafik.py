import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_iou_data(file_path, output_folder):
    # Load the data
    data = pd.read_csv(file_path, delim_whitespace=True, header=None)
    data.columns = ['id', 'iou']
    
    # Sort data by 'id'
    sorted_data = data.sort_values(by='id')
    
    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_data['id'], sorted_data['iou'], marker='o')
    plt.xlabel('Frame ID')
    plt.ylabel('IoU Value')
    plt.title('IoU Values Across Frames')
    plt.grid(True)
    
    # Save the figure to the specified folder
    output_file = os.path.join(output_folder, 'iou_graph.png')  # Change file extension if needed
    plt.savefig(output_file)
    

file_path = '/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/frame_iou.txt'
output_folder = '/Users/abdulbarisoylemez/Documents/VisualCode/yolo-track/iouGrafik'
visualize_iou_data(file_path, output_folder)
